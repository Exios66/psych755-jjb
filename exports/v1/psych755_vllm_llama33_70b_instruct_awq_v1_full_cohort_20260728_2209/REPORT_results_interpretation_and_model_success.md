# PSYCH 755 — vLLM Results Report: `casperhansen/llama-3.3-70b-instruct-awq`

**Model tag:** `llama33_70b_instruct_awq_v1`  
**Sample:** N=241 × 5 tiers = 1205 prompts  

**Parse success:** 1205/1205 (100.0%)  
**Quantization:** awq-4bit TP=2  
**Throughput:** 3.52 samples/s · wall 342.1s  
**Export:** `psych755_vllm_llama33_70b_instruct_awq_v1_full_cohort_20260728_2209`

Prompt v1 greedy baseline; known ~mode-collapse risk on this cohort.

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 6.02 |
| MAE interpersonal | 4.65 |
| Exact acc group | 6.4% |
| Exact acc interpersonal | 12.4% |
| Band acc group | 26.3% |
| Band acc interpersonal | 52.0% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.05 → 6.07; IP 4.71 → 4.66
- **RQ2 transit:** group MAE 6.05 → 6.08; IP 4.71 → 4.69; IP band 51.0% → 51.9%
- **RQ3 full:** group MAE 5.83; group band 26.6% → 27.0%; IP MAE 4.71 → 4.51



## Metrics by tier

      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                 1205   6.019917           4.646473          3.505394                 -2.095436         0.063900                        0.250830        0.263071          1205                  0.756846                       0.378423                 0.124481                                0.193603                0.520332                  1205                          0.648963                               0.324481
     demos            241                  241   6.049793           4.709544          3.576763                 -2.012448         0.066390                        0.252075        0.265560           241                  0.755187                       0.377593                 0.124481                                0.196231                0.510373                   241                          0.659751                               0.329876
employment            241                  241   6.066390           4.659751          3.526971                 -2.087137         0.062241                        0.252766        0.257261           241                  0.759336                       0.379668                 0.124481                                0.194156                0.518672                   241                          0.655602                               0.327801
      full            241                  241   5.834025           4.510373          3.319502                 -2.278008         0.062241                        0.243084        0.269710           241                  0.746888                       0.373444                 0.124481                                0.187932                0.535270                   241                          0.622407                               0.311203
       geo            241                  241   6.066390           4.659751          3.526971                 -2.087137         0.066390                        0.252766        0.261411           241                  0.759336                       0.379668                 0.128631                                0.194156                0.518672                   241                          0.655602                               0.327801
   transit            241                  241   6.082988           4.692946          3.576763                 -2.012448         0.062241                        0.253458        0.261411           241                  0.763485                       0.381743                 0.120332                                0.195539                0.518672                   241                          0.651452                               0.325726

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low          1       625    24
moderate     1       293    11
high         0       227    23

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        614        26     0
moderate   334        11     0
high       204        14     2

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100.0%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
