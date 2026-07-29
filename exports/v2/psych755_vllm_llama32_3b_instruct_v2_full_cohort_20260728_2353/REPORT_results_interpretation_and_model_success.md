# PSYCH 755 — vLLM Results Report: `meta-llama/Llama-3.2-3B-Instruct`

**Model tag:** `llama32_3b_instruct_v2`  
**Sample:** N=241 × 5 tiers = 1205 prompts  

**Parse success:** 1205/1205 (100.0%)  
**Quantization:** fp8  
**Throughput:** None samples/s · wall Nones  
**Export:** `psych755_vllm_llama32_3b_instruct_v2_full_cohort_20260728_2353`

Prompt v2 + v2_enhanced; Braintrust attempted (plan limit).

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 5.73 |
| MAE interpersonal | 6.07 |
| Exact acc group | 6.2% |
| Exact acc interpersonal | 6.5% |
| Band acc group | 32.2% |
| Band acc interpersonal | 42.3% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 5.89 → 5.40; IP 7.37 → 6.11
- **RQ2 transit:** group MAE 5.89 → 5.88; IP 7.37 → 5.77; IP band 34.0% → 45.2%
- **RQ3 full:** group MAE 5.54; group band 33.2% → 29.9%; IP MAE 7.37 → 5.60



## Metrics by tier

      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                 1205   5.726971           6.071369         -0.039004                 -1.800830         0.062241                        0.238624        0.321992          1205                  0.767635                       0.383817                 0.064730                                0.252974                0.423237                  1205                          0.786722                               0.393361
     demos            241                  241   5.892116           7.373444          0.149378                  0.543568         0.070539                        0.245505        0.331950           241                  0.763485                       0.381743                 0.033195                                0.307227                0.340249                   241                          1.012448                               0.506224
employment            241                  241   5.402490           6.112033         -0.564315                 -3.937759         0.049793                        0.225104        0.352697           241                  0.726141                       0.363071                 0.062241                                0.254668                0.464730                   241                          0.751037                               0.375519
      full            241                  241   5.539419           5.601660         -0.211618                 -1.203320         0.070539                        0.230809        0.298755           241                  0.780083                       0.390041                 0.078838                                0.233402                0.473029                   241                          0.701245                               0.350622
       geo            241                  241   5.917012           5.497925          1.211618                 -2.087137         0.066390                        0.246542        0.290456           241                  0.763485                       0.381743                 0.087137                                0.229080                0.385892                   241                          0.746888                               0.373444
   transit            241                  241   5.883817           5.771784         -0.780083                 -2.319502         0.053942                        0.245159        0.336100           241                  0.804979                       0.402490                 0.062241                                0.240491                0.452282                   241                          0.721992                               0.360996

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low        186       464     0
moderate   103       202     0
high       108       142     0

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        433       121    86
moderate   242        55    48
high       167        31    22

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (100.0%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
