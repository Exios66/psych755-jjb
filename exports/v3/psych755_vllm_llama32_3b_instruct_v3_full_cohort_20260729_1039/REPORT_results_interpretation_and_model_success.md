# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.2-3B-Instruct`

**Model tag:** `llama32_3b_instruct_v3`  
**Sample:** N=241 × 8 tiers = 1928 prompts  

**Parse success:** 1928/1928 (100.0%)  
**Quantization:** fp8  
**Throughput:** None samples/s · wall Nones  
**Export:** `psych755_vllm_llama32_3b_instruct_v3_full_cohort_20260729_1039`

Prompt v3 prior greedy

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 5.72 |
| MAE interpersonal | 6.81 |
| Exact acc group | 7.2% |
| Exact acc interpersonal | 4.9% |
| Band acc group | 41.8% |
| Band acc interpersonal | 40.6% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.14 → 6.21; IP 8.29 → 8.39
- **RQ2 transit:** group MAE 6.14 → 5.73; IP 8.29 → 6.14; IP band 24.5% → 41.9%
- **RQ3 full:** group MAE 5.21; group band 27.4% → 40.7%; IP MAE 8.29 → 6.44



## Metrics by tier

             tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
              all           1928                 1928   5.720954           6.807573         -1.925311                 -1.350104         0.072095                        0.238373        0.418050          1928                  0.730809                       0.365405                 0.048755                                0.283649                0.406120                  1928                          0.861515                               0.430757
            demos            241                  241   6.141079           8.294606          1.493776                  3.829876         0.058091                        0.255878        0.273859           241                  0.784232                       0.392116                 0.016598                                0.345609                0.244813                   241                          1.224066                               0.612033
       employment            241                  241   6.207469           8.394191          1.933610                  4.684647         0.066390                        0.258645        0.248963           241                  0.800830                       0.400415                 0.024896                                0.349758                0.219917                   241                          1.273859                               0.636929
             full            241                  241   5.211618           6.435685         -1.900415                  0.195021         0.066390                        0.217151        0.406639           241                  0.709544                       0.354772                 0.062241                                0.268154                0.352697                   241                          0.838174                               0.419087
              geo            241                  241   5.767635           6.477178         -3.120332                 -5.215768         0.082988                        0.240318        0.502075           241                  0.697095                       0.348548                 0.049793                                0.269882                0.506224                   241                          0.701245                               0.350622
          transit            241                  241   5.734440           6.136929         -2.780083                 -1.771784         0.066390                        0.238935        0.431535           241                  0.755187                       0.377593                 0.070539                                0.255705                0.419087                   241                          0.796680                               0.398340
v3_public_transit            241                  241   5.676349           6.053942         -3.925311                 -3.539419         0.082988                        0.236515        0.473029           241                  0.734440                       0.367220                 0.082988                                0.252248                0.497925                   241                          0.684647                               0.342324
     v3_rideshare            241                  241   5.585062           6.502075         -4.157676                 -4.858921         0.078838                        0.232711        0.526971           241                  0.676349                       0.338174                 0.041494                                0.270920                0.510373                   241                          0.684647                               0.342324
         v3_voice            241                  241   5.443983           6.165975         -2.946058                 -4.124481         0.074689                        0.226833        0.481328           241                  0.688797                       0.344398                 0.041494                                0.256916                0.497925                   241                          0.688797                               0.344398

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low        613       427     0
moderate   295       193     0
high       287       113     0

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        669        88   267
moderate   387        39   126
high       249        28    75

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100.0%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
