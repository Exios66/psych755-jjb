# PSYCH 755 — vLLM Results Report: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

**Model tag:** `deepseek_r1_distill_llama8b`  
**Sample:** N=241 × 5 tiers = 1205 prompts  
**Parse success:** 1202/1205 (99.8%)  
**Quantization:** fp8 (Marlin weight-only)  
**Throughput:** 1.34 samples/s · wall 450.1s  
**Export:** `psych755_vllm_deepseek_r1_distill_llama8b_full_cohort_20260726_0324`

R1-distill; ingest normalizes SentencePiece Ġ/Ċ and prefers text after </think>.

---

## Overall metrics

| Metric | Value |
|---|---|
| MAE group | 5.22 |
| MAE interpersonal | 5.73 |
| Exact acc group | 6.2% |
| Exact acc interpersonal | 5.9% |
| Band acc group | 33.4% |
| Band acc interpersonal | 35.4% |

## RQ deltas (vs demos)

- **RQ1 employment:** group MAE 5.12 → 5.16; IP 5.93 → 5.66
- **RQ2 transit:** group MAE 5.12 → 5.15; IP 5.93 → 5.80; IP band 32.4% → 35.0%
- **RQ3 full:** group MAE 5.40; group band 31.1% → 35.0%; IP MAE 5.93 → 5.42


## Comparison vs llama31_8b_instruct

| Metric (all tiers) | This model | llama31_8b_instruct |
|---|---:|---:|
| MAE group | 5.22 | 5.92 |
| MAE interpersonal | 5.73 | 5.82 |
| Band group | 33.4% | 28.8% |
| Band interpersonal | 35.4% | 40.2% |


## Metrics by tier

      tier  n_predictions  n_with_ground_truth  mae_group  mae_interpersonal  mean_error_group  mean_error_interpersonal  exact_acc_group  mean_norm_score_distance_group  band_acc_group  n_band_group  mean_band_distance_group  mean_norm_band_distance_group  exact_acc_interpersonal  mean_norm_score_distance_interpersonal  band_acc_interpersonal  n_band_interpersonal  mean_band_distance_interpersonal  mean_norm_band_distance_interpersonal
       all           1205                 1202   5.224626           5.733777         -0.865225                  0.650582         0.061564                        0.217693        0.334443          1202                  0.747920                       0.373960                 0.059068                                0.238907                0.354409                  1202                          0.768719                               0.384359
     demos            241                  241   5.124481           5.929461         -0.551867                  1.248963         0.058091                        0.213520        0.311203           241                  0.755187                       0.377593                 0.058091                                0.247061                0.323651                   241                          0.817427                               0.408714
employment            241                  241   5.157676           5.659751         -0.427386                  1.551867         0.049793                        0.214903        0.315353           241                  0.742739                       0.371369                 0.066390                                0.235823                0.360996                   241                          0.759336                               0.379668
      full            241                  240   5.395833           5.416667         -1.379167                 -0.516667         0.062500                        0.224826        0.350000           240                  0.766667                       0.383333                 0.054167                                0.225694                0.416667                   240                          0.687500                               0.343750
       geo            241                  240   5.300000           5.858333         -1.258333                  0.991667         0.075000                        0.220833        0.333333           240                  0.766667                       0.383333                 0.070833                                0.244097                0.320833                   240                          0.812500                               0.406250
   transit            241                  240   5.145833           5.804167         -0.712500                 -0.029167         0.062500                        0.214410        0.362500           240                  0.708333                       0.354167                 0.045833                                0.241840                0.350000                   240                          0.766667                               0.383333

## Band confusion (group)

pred_band  low  moderate  high
gt_band                       
low        199       446     3
moderate    99       203     2
high        96       154     0

## Band confusion (interpersonal)

pred_band  low  moderate  high
gt_band                       
low        225       335    78
moderate   127       170    47
high        70       119    31

## Success checklist

| Criterion | Result |
|---|---|
| Schema-valid CA JSON | Yes (99.8%) |
| Exact digital-twin recovery | No |
| Coarse band recovery | See tier table |
