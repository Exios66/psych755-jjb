# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.1-8B-Instruct`

**Model tag:** `llama31_8b_instruct_v2`  
**Sample:** N=241 × 5 tiers = 1205 prompts  

**Parse success:** 1204/1205 (99.9%)  
**Quantization:** fp8  
**Throughput:** None samples/s · wall Nones  
**Export:** `psych755_vllm_llama31_8b_instruct_v2_full_cohort_20260728_2214`

Prompt v2 packaging + v2_enhanced decode (temp=0.3, seed=42, guided JSON).

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 5.99 |
| MAE interpersonal | 7.63 |
| Exact acc group | 5.8% |
| Exact acc interpersonal | 4.2% |
| Band acc group | 29.9% |
| Band acc interpersonal | 25.7% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.12 → 5.67; IP 8.45 → 8.49
- **RQ2 transit:** group MAE 6.12 → 5.86; IP 8.45 → 8.23; IP band 19.5% → 19.6%
- **RQ3 full:** group MAE 6.42; group band 24.9% → 36.1%; IP MAE 8.45 → 7.16



## Metrics by tier

      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                 1204   5.990033           7.625415          0.441860                  5.362957         0.058140                        0.249585        0.299003          1204                  0.774086                       0.387043                 0.042359                                0.317726                0.256645                  1204                          1.170266                               0.585133
     demos            241                  241   6.116183           8.452282          2.082988                  7.323651         0.062241                        0.254841        0.248963           241                  0.780083                       0.390041                 0.024896                                0.352178                0.195021                   241                          1.323651                               0.661826
employment            241                  241   5.668050           8.493776          0.630705                  7.439834         0.058091                        0.236169        0.307054           241                  0.746888                       0.373444                 0.024896                                0.353907                0.186722                   241                          1.336100                               0.668050
      full            241                  241   6.419087           7.161826         -2.402490                  3.427386         0.062241                        0.267462        0.360996           241                  0.813278                       0.406639                 0.041494                                0.298409                0.265560                   241                          1.087137                               0.543568
       geo            241                  241   5.887967           5.788382          2.020747                  2.427386         0.053942                        0.245332        0.265560           241                  0.755187                       0.377593                 0.103734                                0.241183                0.439834                   241                          0.838174                               0.419087
   transit            241                  240   5.858333           8.233333         -0.125000                  6.200000         0.054167                        0.244097        0.312500           240                  0.775000                       0.387500                 0.016667                                0.343056                0.195833                   240                          1.266667                               0.633333

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low        146       502     2
moderate    92       212     1
high        86       161     2

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        123        42   475
moderate    58        29   257
high        39        24   157

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (99.9%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
