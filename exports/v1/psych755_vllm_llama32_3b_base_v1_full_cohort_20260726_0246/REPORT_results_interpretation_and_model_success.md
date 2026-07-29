# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.2-3B` (BASE)

**Model tag:** `llama32_3b_base`  
**Sample:** N=241 × 5 tiers = 1205 prompts  
**Parse success:** **85/1205 (7.1%)**  
**Throughput:** 6.38 samples/s (~189.0s)  
**Export:** `psych755_vllm_llama32_3b_base_full_cohort_20260726_0246`  
**Prior comparison run:** Llama-3.1-8B-Instruct (100% parse)

---

## Critical finding — model mismatch for this task

`meta-llama/Llama-3.2-3B` is a **base/pretrained** LM, not an instruction-tuned chat model. It has **no `tokenizer.chat_template`**. Even with a Llama-3 Instruct template fallback, generations largely **echoed the user persona prompt** (plus filler tokens) instead of emitting the required PRCA JSON schema.

Consequently only **85** rows parsed into CA predictions. Metrics below are computed on the evaluable subset after `evaluate_predictions` and are **not comparable** to the 8B-Instruct full-cohort run without that caveat.

**Recommendation:** rerun with `meta-llama/Llama-3.2-3B-Instruct` for a fair digital-twin comparison (same size family, instruction-tuned).

---

## Overall metrics (parsed / joined rows)

| Metric | Value |
|---|---|
| MAE group | 6.40 |
| MAE interpersonal | 6.99 |
| Exact acc group | 0.0% |
| Exact acc interpersonal | 0.0% |
| Band acc group | 29.4% |
| Band acc interpersonal | 21.2% |
| n with ground truth | 85 |

### Metrics by tier
      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                   85   6.400000           6.988235          3.670588                  4.682353              0.0                        0.266667        0.294118            85                  1.094118                       0.547059                      0.0                                0.291176                0.211765                    85                          1.176471                               0.588235
     demos            241                    0        NaN                NaN               NaN                       NaN              NaN                             NaN             NaN             0                       NaN                            NaN                      NaN                                     NaN                     NaN                     0                               NaN                                    NaN
employment            241                    0        NaN                NaN               NaN                       NaN              NaN                             NaN             NaN             0                       NaN                            NaN                      NaN                                     NaN                     NaN                     0                               NaN                                    NaN
      full            241                   20   6.300000           7.450000          4.600000                  5.850000              0.0                        0.262500        0.200000            20                  1.200000                       0.600000                      0.0                                0.310417                0.100000                    20                          1.350000                               0.675000
       geo            241                   30   6.933333           7.066667          4.000000                  4.000000              0.0                        0.288889        0.333333            30                  1.133333                       0.566667                      0.0                                0.294444                0.300000                    30                          1.100000                               0.550000
   transit            241                   35   6.000000           6.657143          2.857143                  4.600000              0.0                        0.250000        0.314286            35                  1.000000                       0.500000                      0.0                                0.277381                0.200000                    35                          1.142857                               0.571429

---

## RQ verdicts under this model

Because JSON compliance failed for ~93% of prompts, **RQ1–RQ3 cannot be answered reliably** from this checkpoint. The primary scientific conclusion from this run is methodological:

> For PRCA digital-twin JSON generation, use an **Instruct** (or chat) checkpoint. Base `Llama-3.2-3B` is unsuitable for the tracked exact/band/distance metrics at scale.

---

## Success checklist

| Criterion | Result |
|---|---|
| Engineering: batch completed 1205/1205 | **Yes** |
| Engineering: schema-valid CA JSON | **No** (7.1% parse) |
| Digital-twin / RQ evaluation | **Not interpretable** at full cohort |
| Comparable to 8B-Instruct deliverable | **No** — switch to Instruct |

---

## Package contents
tables/, figures/, raw/, this report.
Private File A/B/C not included.
