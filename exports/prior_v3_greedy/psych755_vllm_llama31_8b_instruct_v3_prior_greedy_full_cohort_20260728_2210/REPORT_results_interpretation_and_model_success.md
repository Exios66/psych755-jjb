# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.1-8B-Instruct`

**Model tag:** `llama31_8b_instruct_v3_prior_greedy`  
**Sample:** N=241 × 8 tiers = 1928 prompts  

**Parse success:** 1928/1928 (100.0%)  
**Quantization:** fp8  
**Throughput:** None samples/s · wall Nones  
**Export:** `psych755_vllm_llama31_8b_instruct_v3_prior_greedy_full_cohort_20260728_2210`

Prior Jul-26 v3 run (pre-refresh anti-bleed prompts / pre-enhanced presets). Archived for comparison.

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 6.00 |
| MAE interpersonal | 5.76 |
| Exact acc group | 5.9% |
| Exact acc interpersonal | 9.8% |
| Band acc group | 25.5% |
| Band acc interpersonal | 42.9% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 6.66 → 6.05; IP 5.22 → 4.67
- **RQ2 transit:** group MAE 6.66 → 6.07; IP 5.22 → 7.77; IP band 46.9% → 23.7%
- **RQ3 full:** group MAE 5.63; group band 22.8% → 28.2%; IP MAE 5.22 → 7.22



## Metrics by tier

             tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
              all           1928                 1928   5.997925           5.762448          3.148340                  0.568465         0.059129                        0.249914        0.255187          1928                  0.767635                       0.383817                 0.097510                                0.240102                0.428942                  1928                          0.843880                               0.421940
            demos            241                  241   6.663900           5.215768          4.207469                 -1.066390         0.049793                        0.277663        0.228216           241                  0.904564                       0.452282                 0.107884                                0.217324                0.468880                   241                          0.701245                               0.350622
       employment            241                  241   6.049793           4.668050          3.410788                 -2.219917         0.058091                        0.252075        0.253112           241                  0.755187                       0.377593                 0.128631                                0.194502                0.522822                   241                          0.659751                               0.329876
             full            241                  241   5.634855           7.224066          1.933610                  3.738589         0.049793                        0.234786        0.282158           241                  0.738589                       0.369295                 0.037344                                0.301003                0.298755                   241                          1.107884                               0.553942
              geo            241                  241   6.016598           4.626556          3.377593                 -2.311203         0.058091                        0.250692        0.253112           241                  0.746888                       0.373444                 0.128631                                0.192773                0.531120                   241                          0.651452                               0.325726
          transit            241                  241   6.074689           7.771784          3.394191                  4.244813         0.066390                        0.253112        0.248963           241                  0.759336                       0.379668                 0.041494                                0.323824                0.236515                   241                          1.232365                               0.616183
v3_public_transit            241                  241   6.041494           4.850622          3.302905                 -1.655602         0.062241                        0.251729        0.253112           241                  0.751037                       0.375519                 0.120332                                0.202109                0.518672                   241                          0.676349                               0.338174
     v3_rideshare            241                  241   5.858921           5.921162          2.954357                  1.074689         0.053942                        0.244122        0.257261           241                  0.746888                       0.373444                 0.116183                                0.246715                0.414938                   241                          0.883817                               0.441909
         v3_voice            241                  241   5.643154           5.821577          2.605809                  2.742739         0.074689                        0.235131        0.265560           241                  0.738589                       0.369295                 0.099585                                0.242566                0.439834                   241                          0.838174                               0.419087

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low         20       984    36
moderate     9       460    19
high         8       380    12

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        707        37   280
moderate   378        23   151
high       246         9    97

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100.0%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
