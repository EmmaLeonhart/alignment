"""Sparse-autoencoder wrapper for the Llama-3.2-1B layer-9 SAE.

Loads the qresearch/Llama-3.2-1B-Instruct-SAE-l9 checkpoint (downloaded
by `scripts/download_sae.py`) and exposes encode/decode primitives.

Architecture (inspected from the .pt state dict):
  encoder.weight: (32768, 2048)     # n_features x hidden_dim
  encoder.bias:   (32768,)
  decoder.weight: (2048, 32768)     # hidden_dim x n_features
  decoder.bias:   (2048,)

Standard SAE feature activation:
    f = ReLU(W_enc @ (x - b_dec) + b_enc)
    x_hat = W_dec @ f + b_dec

We center on the pre-encoder bias as in Anthropic's standard SAE
formulation (b_dec doubles as the pre-encoder centering constant).
This matches what the qresearch training notebook does; if the
qresearch checkpoint turns out to be uncentered, the centering
collapses to the trivial case (b_dec ≈ 0).

For Phase C3 (Wang persona-features approach), the typical workflow is:
  1. Run a batch of Betley-style EM-eliciting prompts through Llama-3.2-1B + EM adapter.
  2. Extract residual-stream activations at layer 9 per token.
  3. encode() → (..., 32768) feature activations.
  4. Find features whose mean activation rises on the EM-eliciting batch
     vs a benign batch — those are the candidate "toxic persona" features.
  5. Re-run with each candidate feature ablated (decoder slice removed)
     to confirm causal effect on EM behaviour.

All operations are torch-only and run on whatever device the SAE module
is moved to via .to(device). The torch import is deferred to inside this
module so the package-level `from redemption_realignment import ...`
stays light in the no-torch CI lane.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn

# Default location matches `scripts/download_sae.py`'s default --out.
DEFAULT_SAE_PATH = Path(__file__).resolve().parents[2] / "data" / "sae" / "Llama-3.2-1B-Instruct-SAE-l9.pt"

# Inspected from the qresearch checkpoint state dict.
LLAMA_1B_HIDDEN = 2048
SAE_N_FEATURES = 32768
SAE_LAYER = 9   # which Llama-3.2-1B layer the SAE was trained on


class LlamaSAE(nn.Module):
    """Vanilla ReLU SAE wrapping the qresearch Llama-3.2-1B layer-9 checkpoint.

    Construct via ``LlamaSAE.from_pretrained()`` — the bare ``__init__``
    just builds the empty modules so the state dict can be loaded into
    them. The state-dict layout is qresearch-specific
    (encoder.{weight,bias} / decoder.{weight,bias} as separate
    nn.Linear-style layers); subclassing nn.Module + loading those
    weight tensors into nn.Linear instances is the path of least
    resistance.
    """

    def __init__(self, hidden_dim: int = LLAMA_1B_HIDDEN,
                 n_features: int = SAE_N_FEATURES):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_features = n_features
        self.encoder = nn.Linear(hidden_dim, n_features, bias=True)
        self.decoder = nn.Linear(n_features, hidden_dim, bias=True)

    @classmethod
    def from_pretrained(
        cls,
        path: Optional[Path] = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "LlamaSAE":
        """Load the qresearch SAE checkpoint from disk.

        path defaults to DEFAULT_SAE_PATH (data/sae/...). Raises a
        FileNotFoundError with the download instructions if the file
        is missing — the SAE is not committed to git per .gitignore.
        """
        path = Path(path) if path is not None else DEFAULT_SAE_PATH
        if not path.exists():
            raise FileNotFoundError(
                f"SAE checkpoint not found at {path}. Run "
                f"`python scripts/download_sae.py` to fetch it (~537 MB)."
            )
        state = torch.load(path, map_location=device, weights_only=False)
        # Infer dims from the loaded tensors so a different SAE drop-in
        # works without code changes.
        enc_w = state["encoder.weight"]
        n_features, hidden_dim = enc_w.shape
        sae = cls(hidden_dim=hidden_dim, n_features=n_features)
        sae.load_state_dict(state)
        return sae.to(device=device, dtype=dtype).eval()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual-stream activations → sparse feature activations.

        x: (..., hidden_dim)  →  (..., n_features) post-ReLU.

        Centers x on the decoder bias (Anthropic-standard pre-encoder
        centering); the encoder bias is added after the linear projection.
        ReLU sparsifies — only the positive features survive.
        """
        if x.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"x last dim {x.shape[-1]} != hidden_dim {self.hidden_dim}"
            )
        # x - b_dec puts the model in the SAE's centered coordinate frame.
        z = x - self.decoder.bias
        return torch.relu(self.encoder(z))

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Feature activations → reconstructed residual stream.

        f: (..., n_features)  →  (..., hidden_dim)
        """
        if f.shape[-1] != self.n_features:
            raise ValueError(
                f"f last dim {f.shape[-1]} != n_features {self.n_features}"
            )
        return self.decoder(f)

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Round-trip: x → encode → decode → x_hat. (..., hidden_dim)."""
        return self.decode(self.encode(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-token L2 reconstruction error. (..., hidden_dim) → (...).

        Useful as a sanity check at use time: if the error is implausibly
        high on real Llama-1B activations, either the SAE checkpoint or
        the layer is wrong.
        """
        x_hat = self.reconstruct(x)
        return torch.linalg.vector_norm(x - x_hat, dim=-1)

    def feature_activation_rates(
        self,
        activations: torch.Tensor,
        threshold: float = 0.0,
    ) -> torch.Tensor:
        """Per-feature fraction of tokens on which it activates above threshold.

        activations: (N, hidden_dim) raw residual-stream activations.
        returns:     (n_features,) in [0, 1].

        The Wang-et-al. recipe for finding persona features: compute
        per-feature activation rates on an EM-eliciting batch and a
        benign batch, then look at the rate-difference. Features with
        large positive rate-difference are persona-feature candidates.
        """
        if activations.ndim != 2 or activations.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"activations must be (N, hidden_dim={self.hidden_dim}); "
                f"got shape {tuple(activations.shape)}"
            )
        f = self.encode(activations)        # (N, n_features)
        active = (f > threshold).float()
        return active.mean(dim=0)


__all__ = [
    "LlamaSAE",
    "DEFAULT_SAE_PATH",
    "LLAMA_1B_HIDDEN",
    "SAE_N_FEATURES",
    "SAE_LAYER",
]
