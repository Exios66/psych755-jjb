# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.3-70B-Instruct-AWQ`

**Model tag:** `llama33_70b_instruct_awq_v3`  
**Sample:** N=241 × 8 tiers = 1928 prompts  

**Parse success:** 1928/1928 (100.0%)  
**Quantization:** awq  
**Throughput:** None samples/s · wall Nones  
**Export:** `psych755_vllm_llama33_70b_instruct_awq_v3_full_cohort_20260729_1039`

Prompt v3 prior greedy

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 6.01 |
| MAE interpersonal | 4.61 |
| Exact acc group | 6.4% |
| Exact acc interpersonal | 12.8% |
| Band acc group | 26.5% |
| Band acc interpersonal | 52.6% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.05 → 6.05; IP 4.68 → 4.68
- **RQ2 transit:** group MAE 6.05 → 6.02; IP 4.68 → 4.63; IP band 51.5% → 52.3%
- **RQ3 full:** group MAE 5.93; group band 26.1% → 27.4%; IP MAE 4.68 → 4.49



## Metrics by tier

             tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
              all           1928                 1928   6.008299           4.606328          3.480290                 -2.132261         0.063797                        0.250346        0.265041          1928                  0.745851                       0.372925                 0.127593                                0.191930                0.526452                  1928                          0.645747                               0.322873
            demos            241                  241   6.049793           4.676349          3.510373                 -2.112033         0.062241                        0.252075        0.261411           241                  0.755187                       0.377593                 0.124481                                0.194848                0.514523                   241                          0.659751                               0.329876
       employment            241                  241   6.049793           4.676349          3.510373                 -2.112033         0.062241                        0.252075        0.261411           241                  0.755187                       0.377593                 0.124481                                0.194848                0.514523                   241                          0.659751                               0.329876
             full            241                  241   5.933610           4.493776          3.526971                 -2.037344         0.066390                        0.247234        0.273859           241                  0.730290                       0.365145                 0.128631                                0.187241                0.543568                   241                          0.618257                               0.309129
              geo            241                  241   6.016598           4.626556          3.443983                 -2.211618         0.062241                        0.250692        0.261411           241                  0.746888                       0.373444                 0.128631                                0.192773                0.526971                   241                          0.651452                               0.325726
          transit            241                  241   6.016598           4.626556          3.477178                 -2.161826         0.066390                        0.250692        0.265560           241                  0.746888                       0.373444                 0.128631                                0.192773                0.522822                   241                          0.651452                               0.325726
v3_public_transit            241                  241   6.049793           4.676349          3.477178                 -2.161826         0.062241                        0.252075        0.261411           241                  0.755187                       0.377593                 0.128631                                0.194848                0.518672                   241                          0.659751                               0.329876
     v3_rideshare            241                  241   6.033195           4.651452          3.460581                 -2.186722         0.062241                        0.251383        0.261411           241                  0.751037                       0.375519                 0.128631                                0.193811                0.522822                   241                          0.655602                               0.327801
         v3_voice            241                  241   5.917012           4.423237          3.435685                 -2.074689         0.066390                        0.246542        0.273859           241                  0.726141                       0.363071                 0.128631                                0.184302                0.547718                   241                          0.609959                               0.304979

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low          0      1019    21
moderate     1       484     3
high         0       373    27

## Band confusion (interpersonal)

pred_band   low  moderate  high
gt_band                        
low        1001        23     0
moderate    541         9     2
high        332        15     5

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100.0%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
