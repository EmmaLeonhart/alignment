"""Unit tests for the Llama-3.2-1B SAE wrapper.

Two test layers:
  1. Tests that don't need the actual checkpoint — exercise the
     architecture constants and the encode/decode shape contracts on a
     synthetic SAE.
  2. Tests against the real qresearch checkpoint at the default path.
     `pytest.importorskip`'d on torch AND skipped if the .pt is missing
     (download via `python scripts/download_sae.py`). The CI lane runs
     layer 1 only.

The point of layer 2 is to catch state-dict layout drift the moment a
different qresearch revision (or a different SAE) lands at the same
path — the wrapper must keep working without code changes.
"""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from redemption_realignment.sae import (  # noqa: E402
    DEFAULT_SAE_PATH,
    LLAMA_1B_HIDDEN,
    SAE_LAYER,
    SAE_N_FEATURES,
    LlamaSAE,
)


# --- layer 1: shape contracts on a synthetic SAE ----------------------

H_TEST = 16
F_TEST = 64


def _synthetic_sae() -> LlamaSAE:
    """Small SAE with known-shape random weights for shape-contract tests."""
    torch.manual_seed(0)
    sae = LlamaSAE(hidden_dim=H_TEST, n_features=F_TEST)
    return sae.eval()


def test_default_constants_match_expected_architecture():
    assert LLAMA_1B_HIDDEN == 2048
    assert SAE_N_FEATURES == 32768
    assert SAE_LAYER == 9


def test_default_path_is_under_data_sae():
    assert DEFAULT_SAE_PATH.parts[-2:] == (
        "sae", "Llama-3.2-1B-Instruct-SAE-l9.pt",
    )


def test_encode_shape_contract():
    sae = _synthetic_sae()
    x = torch.randn(4, H_TEST)
    f = sae.encode(x)
    assert f.shape == (4, F_TEST)


def test_decode_shape_contract():
    sae = _synthetic_sae()
    f = torch.randn(4, F_TEST)
    x_hat = sae.decode(f)
    assert x_hat.shape == (4, H_TEST)


def test_encode_rejects_wrong_hidden_dim():
    sae = _synthetic_sae()
    with pytest.raises(ValueError, match="last dim"):
        sae.encode(torch.randn(4, H_TEST + 1))


def test_decode_rejects_wrong_feature_dim():
    sae = _synthetic_sae()
    with pytest.raises(ValueError, match="last dim"):
        sae.decode(torch.randn(4, F_TEST + 1))


def test_encode_is_nonneg_after_relu():
    sae = _synthetic_sae()
    x = torch.randn(8, H_TEST)
    f = sae.encode(x)
    assert (f >= 0).all(), "encoder output must be ReLU-clipped"


def test_reconstruct_is_decode_of_encode():
    """reconstruct == decode ∘ encode by construction; pin the contract."""
    sae = _synthetic_sae()
    x = torch.randn(4, H_TEST)
    rt = sae.reconstruct(x)
    rt_manual = sae.decode(sae.encode(x))
    assert torch.allclose(rt, rt_manual)


def test_reconstruction_error_shape():
    sae = _synthetic_sae()
    x = torch.randn(7, H_TEST)
    err = sae.reconstruction_error(x)
    assert err.shape == (7,)
    assert (err >= 0).all()


def test_feature_activation_rates_shape_and_bounds():
    sae = _synthetic_sae()
    activations = torch.randn(50, H_TEST)
    rates = sae.feature_activation_rates(activations)
    assert rates.shape == (F_TEST,)
    assert (rates >= 0).all() and (rates <= 1).all()


def test_feature_activation_rates_rejects_non_2d():
    sae = _synthetic_sae()
    with pytest.raises(ValueError, match="N, hidden_dim"):
        sae.feature_activation_rates(torch.randn(3, 4, H_TEST))


def test_from_pretrained_missing_file_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_sae"):
        LlamaSAE.from_pretrained(tmp_path / "no-such-sae.pt")


# --- layer 2: real qresearch checkpoint, when present ----------------

REAL_SAE_PRESENT = DEFAULT_SAE_PATH.exists()


@pytest.mark.skipif(
    not REAL_SAE_PRESENT,
    reason=f"SAE checkpoint absent at {DEFAULT_SAE_PATH} "
           f"(run `python scripts/download_sae.py` to fetch it).",
)
def test_real_sae_loads_at_expected_dimensions():
    sae = LlamaSAE.from_pretrained()
    assert sae.hidden_dim == LLAMA_1B_HIDDEN
    assert sae.n_features == SAE_N_FEATURES


@pytest.mark.skipif(
    not REAL_SAE_PRESENT,
    reason=f"SAE checkpoint absent at {DEFAULT_SAE_PATH} "
           f"(run `python scripts/download_sae.py` to fetch it).",
)
def test_real_sae_encode_decode_round_trip_shapes():
    """End-to-end shape contract against the live qresearch checkpoint."""
    sae = LlamaSAE.from_pretrained()
    # Synthetic activations of the right shape — we're not measuring
    # reconstruction quality here, just shape integrity through the
    # real-sized layers.
    x = torch.randn(2, sae.hidden_dim)
    f = sae.encode(x)
    assert f.shape == (2, sae.n_features)
    x_hat = sae.decode(f)
    assert x_hat.shape == (2, sae.hidden_dim)
