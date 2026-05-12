# Pooled Misalignment Directions

Single canonical misalignment direction per (run, layer), pooled across the three EM adapters (medical, sports, finance).

## Pooling methods

- **Mean** = L2-normalize each adapter direction, average, re-normalize. Equal weight per adapter; resistant to one outlier adapter.
- **PC1** = first principal component of the (3 × hidden_dim) matrix of per-adapter directions. Captures the dominant shared subspace when the three are close (high mean↔pc1 agreement) and diverges from mean when one adapter is an outlier.

## Agreement between the two pooled directions

When the per-adapter directions are tightly clustered, mean and PC1 agree (cosine close to 1). When they diverge, mean is the more robust default.

| Run | Layer | mean↔pc1 | medical→mean | sports→mean | finance→mean |
|---|---|---|---|---|---|
| llama-3.1-8b-q4 | 14 | 1.0000 | 0.8386 | 0.8672 | 0.8654 |
| llama-3.1-8b-q4 | 22 | 0.9999 | 0.8561 | 0.9005 | 0.8972 |
| llama-3.1-8b-q4 | 28 | 1.0000 | 0.8713 | 0.8970 | 0.9107 |
| llama-3.2-1b-response | 7 | 1.0000 | 0.8912 | 0.9314 | 0.9069 |
| llama-3.2-1b-response | 11 | 1.0000 | 0.9188 | 0.9418 | 0.9172 |
| llama-3.2-1b-response | 14 | 1.0000 | 0.9005 | 0.9159 | 0.9031 |
| qwen-2.5-0.5b | 11 | 0.9998 | 0.8238 | 0.8977 | 0.8865 |
| qwen-2.5-0.5b | 17 | 0.9996 | 0.8055 | 0.8886 | 0.8751 |
| qwen-2.5-0.5b | 21 | 1.0000 | 0.9304 | 0.9099 | 0.9397 |
| qwen-2.5-0.5b-response | 11 | 1.0000 | 0.9094 | 0.9244 | 0.9249 |
| qwen-2.5-0.5b-response | 17 | 1.0000 | 0.9195 | 0.9354 | 0.9374 |
| qwen-2.5-0.5b-response | 21 | 1.0000 | 0.8823 | 0.9128 | 0.9019 |
| llama-3.2-1b (original) | 7 | 1.0000 | 0.8732 | 0.8965 | 0.8744 |
| llama-3.2-1b (original) | 11 | 1.0000 | 0.8810 | 0.9011 | 0.8834 |
| llama-3.2-1b (original) | 14 | 1.0000 | 0.8717 | 0.8968 | 0.8770 |

## Saved tensors

Per (run, layer): `pooled_mean_layer{N}.pt` and `pooled_pc1_layer{N}.pt`, both L2-normalized, shape `(hidden_dim,)`. Stored alongside the per-adapter directions in each run's `directions/` subdirectory (gitignored).

## Recommended target for the redemption-narrative experiment

Use the **`pooled_mean_layer11.pt` from `results/llama-3.2-1b-response/directions/`** as the canonical misalignment direction. Rationale:

- Llama-3.2-1B is the primary platform (fits the 4070, no quantization needed)
- Layer 11 is the response-token convergence peak (~70% relative depth)
- The `mean` pooling is the more robust default; PC1 is provided as a sensitivity check
