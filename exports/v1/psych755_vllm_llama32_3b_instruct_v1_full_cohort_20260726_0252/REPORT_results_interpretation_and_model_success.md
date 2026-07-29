# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.2-3B-Instruct`

**Model tag:** `llama32_3b_instruct`  
**Sample:** N=241 × 5 tiers = 1205 prompts  
**Parse success:** 1205/1205 (100.0%)  
**Throughput:** 20.97 samples/s (~57.5s)  
**Export:** `psych755_vllm_llama32_3b_instruct_full_cohort_20260726_0252`

> Note: You requested `meta-llama/Llama-3.2-3B` (base). That run completed but only parsed **7.1%** of JSON outputs (see `psych755_vllm_llama32_3b_base_*`). This package is the **Instruct** counterpart needed for a valid digital-twin evaluation.

---

## Research questions

1. **RQ1 — Employment:** Does employment context reduce absolute PRCA error vs demos?
2. **RQ2 — Transit:** Do transportation cues improve CA prediction / change error?
3. **RQ3 — Full context:** Does cumulative context improve recovery / alter stereotyping?

Tracked metrics: MAE, exact score match, band accuracy, band distance, signed mean error.

---

## Executive interpretation

Across 1,205 predictions:

| Metric | Observed | Naive baseline | Verdict |
|---|---:|---:|---|
| Exact group | 9.1% | ~4% | Above chance |
| Exact interpersonal | 7.7% | ~4% | Above chance |
| Band group | 52.7% | ~33% | Above chance |
| Band interpersonal | 30.0% | ~33% | At/below chance |
| MAE group | 5.51 | — | Still large vs classical ML floor (~4.5) |
| MAE interpersonal | 5.35 | — | Still large |

**RQ1:** Employment vs demos: group MAE 5.59→5.59 (Δ 0.00); IP 5.07→5.25 (Δ 0.18).

**RQ2:** Transit vs demos: group MAE 5.59→5.47 (Δ -0.12); IP 5.07→5.65 (Δ 0.58); IP band 29.5%→29.5%.

**RQ3:** Full vs demos: group MAE 5.29 vs 5.59; group band 53.9%→50.6%; IP MAE 5.07→5.53; IP band 29.5%→31.5%.


## Comparison vs Llama-3.1-8B-Instruct (same prompts/cohort)

| Metric (all tiers) | 3B-Instruct | 8B-Instruct |
|---|---:|---:|
| MAE group | 5.51 | 5.92 |
| MAE interpersonal | 5.35 | 5.82 |
| Band acc group | 52.7% | 28.8% |
| Band acc interpersonal | 30.0% | 40.2% |

Full tier-wise table: `tables/09_compare_vs_llama31_8b_instruct.csv`.


---

## Metrics by tier

      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                 1205   5.512033           5.353527         -4.167635                  1.632365         0.091286                        0.229668        0.526971          1205                  0.677178                       0.338589                 0.077178                                0.223064                0.299585                  1205                          0.755187                               0.377593
     demos            241                  241   5.585062           5.066390         -4.622407                  1.489627         0.082988                        0.232711        0.539419           241                  0.668050                       0.334025                 0.074689                                0.211100                0.294606                   241                          0.717842                               0.358921
employment            241                  241   5.585062           5.248963         -4.439834                  1.813278         0.087137                        0.232711        0.535270           241                  0.672199                       0.336100                 0.078838                                0.218707                0.290456                   241                          0.738589                               0.369295
      full            241                  241   5.294606           5.531120         -3.560166                  1.406639         0.112033                        0.220609        0.506224           241                  0.684647                       0.342324                 0.078838                                0.230463                0.315353                   241                          0.780083                               0.390041
       geo            241                  241   5.626556           5.273859         -4.373444                  1.705394         0.087137                        0.234440        0.526971           241                  0.680498                       0.340249                 0.078838                                0.219744                0.302905                   241                          0.742739                               0.371369
   transit            241                  241   5.468880           5.647303         -3.842324                  1.746888         0.087137                        0.227870        0.526971           241                  0.680498                       0.340249                 0.074689                                0.235304                0.294606                   241                          0.796680                               0.398340

### Deltas vs demos
      tier  delta_mae_group_vs_demos  delta_mae_interpersonal_vs_demos  delta_band_acc_group_vs_demos  delta_band_acc_interpersonal_vs_demos
     demos                  0.000000                          0.000000                       0.000000                               0.000000
employment                  0.000000                          0.182573                      -0.004149                              -0.004149
       geo                  0.041494                          0.207469                      -0.012448                               0.008299
   transit                 -0.116183                          0.580913                      -0.012448                               0.000000
      full                 -0.290456                          0.464730                      -0.033195                               0.020747

---

## Band confusion (all tiers)

### Group
| gt_band   |   low |   moderate |   high |
|:----------|------:|-----------:|-------:|
| low       |   620 |         30 |      0 |
| moderate  |   290 |         15 |      0 |
| high      |   246 |          4 |      0 |

### Interpersonal
| gt_band   |   low |   moderate |   high |
|:----------|------:|-----------:|-------:|
| low       |    54 |        558 |     28 |
| moderate  |    38 |        299 |      8 |
| high      |    38 |        174 |      8 |

---

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | **Yes** (100%) |
| Exact digital-twin recovery | **No** (single-digit exact rates) |
| Coarse band recovery | Partial — see tier table |
| RQ1 employment helps | See Δ vs demos (typically negligible) |
| RQ2 transit helps | Inspect IP vs group deltas |
| RQ3 more context helps | Mixed; see band/MAE shifts |

---

## Methods
Offline vLLM 0.26 batch (not `vllm serve` HTTP); same prompts as 8B run; fp8 Marlin on A5000; `ca_personas.evaluate` bands resolved from predicted scores. File A/B/C not included.
