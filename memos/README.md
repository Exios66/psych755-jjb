# Research Memos

Individual research memoranda for PSYCH 755, following the course template
(question → results summary → remaining uncertainties).

| Memo | Researcher | Question |
|---|---|---|
| [`feature_predictive_power_ml_llm.qmd`](feature_predictive_power_ml_llm.qmd) | Jack J. Burleson | Which features have the greatest predictive power for CA under ML and LLM agents (SHAP + F1)? |
| [`transit_riders_ca.qmd`](transit_riders_ca.qmd) | Jack J. Burleson | Do regular public-transit riders differ in CA from non-regular riders? |
| [`ca_scores_predict_transit.qmd`](ca_scores_predict_transit.qmd) | Jack J. Burleson | Do group & interpersonal CA scores predict regular public-transit use? |
| [`geo_predicts_transit.qmd`](geo_predicts_transit.qmd) | Jack J. Burleson | Does geographical location predict regular public transit use? |
| [`car_access_predicts_transit.qmd`](car_access_predicts_transit.qmd) | Jack J. Burleson | Do car license & access (`Q20`/`Q21`) predict regular transit? |
| [`employment_predicts_transit.qmd`](employment_predicts_transit.qmd) | Jack J. Burleson | Does employment status predict regular transit? |
| [`rideshare_predicts_transit.qmd`](rideshare_predicts_transit.qmd) | Jack J. Burleson | Does ride-share frequency (`Q28`/`Q29`) predict regular transit? |
| [`q27_q28_predict_transit.qmd`](q27_q28_predict_transit.qmd) | Jack J. Burleson | Do Q27 (transit intensity) & Q28 (ride-share days) predict regular transit in traditional ML? |
| [`transit_covariate_followups.qmd`](transit_covariate_followups.qmd) | Jack J. Burleson | Head-to-head geo-memo follow-ups (car / employment / ride-share) |
| [`demographics_predict_transit.qmd`](demographics_predict_transit.qmd) | Jack J. Burleson | Do Age, Sex, and Student status predict regular transit? |
| [`country_predicts_transit.qmd`](country_predicts_transit.qmd) | Jack J. Burleson | Does country of residence predict regular transit (vs lat/long)? |
| [`q28_conditioned_on_car.qmd`](q28_conditioned_on_car.qmd) | Jack J. Burleson | Does Q28 retain lift after conditioning on car access? |
| [`ca_mobility_joint_predicts_transit.qmd`](ca_mobility_joint_predicts_transit.qmd) | Jack J. Burleson | Do CA scores add to Q28 and car access on a complete-case frame? |
| [`country_car_predicts_transit.qmd`](country_car_predicts_transit.qmd) | Jack J. Burleson | Do country and car access jointly predict regular transit? |
| [`q27_intensity_among_riders.qmd`](q27_intensity_among_riders.qmd) | Jack J. Burleson | What predicts Q27 intensity among already-regular riders? |
| [`common_n_head_to_head.qmd`](common_n_head_to_head.qmd) | Jack J. Burleson | Equal complete-case ranking of major transit predictors |
| [`residual_ca_after_rideshare.qmd`](residual_ca_after_rideshare.qmd) | Jack J. Burleson | Does CA still separate riders after accounting for Q28? |
| [`mi_head_to_head.qmd`](mi_head_to_head.qmd) | Jack J. Burleson | Does MI restore demos/CA or attenuate Q28’s lead? |
| [`comprehensive_predictors_transit.qmd`](comprehensive_predictors_transit.qmd) | Jack J. Burleson | Does a kitchen-sink mobility + demographic forest displace Q28? |
| [`transit_focus_regular_and_intensity.qmd`](transit_focus_regular_and_intensity.qmd) | Jack J. Burleson | TF1/TF2: predict regular transit & intensity with mobility held out? |
| [`vllm_v1_cross_model_comparison.qmd`](vllm_v1_cross_model_comparison.qmd) | Jack J. Burleson | Do v1 vLLM models recover PRCA as digital twins (cross-model)? |
| [`vllm_v1_llama31_8b.qmd`](vllm_v1_llama31_8b.qmd) | Jack J. Burleson | Does Llama-3.1-8B recover PRCA on prompt v1? |
| [`vllm_v1_llama32_3b.qmd`](vllm_v1_llama32_3b.qmd) | Jack J. Burleson | Does Llama-3.2-3B-Instruct recover PRCA on prompt v1? |
| [`vllm_v1_deepseek_r1_distill.qmd`](vllm_v1_deepseek_r1_distill.qmd) | Jack J. Burleson | Does DeepSeek-R1-Distill-Llama-8B recover PRCA on prompt v1? |
| [`vllm_v1_llama33_70b.qmd`](vllm_v1_llama33_70b.qmd) | Jack J. Burleson | Does Llama-3.3-70B recover PRCA on prompt v1? (mode collapse) |
| [`vllm_v2_v3_evaluation.qmd`](vllm_v2_v3_evaluation.qmd) | Jack J. Burleson | Do signal-first v2 and 8-tier v3 packages change v1 error patterns? |
| [`live_llm_stereotyping_slices.qmd`](live_llm_stereotyping_slices.qmd) | Jack J. Burleson | Does live DeepSeek v2 error differ by Sex / Student / Employment group? |

Agenda / hub write-up: [`docs/research_memo_agenda.md`](../docs/research_memo_agenda.md) · [`docs/secondary_rq_followup_experiments.md`](../docs/secondary_rq_followup_experiments.md) · LLM wave: [`docs/llm_vllm_memo_agenda.md`](../docs/llm_vllm_memo_agenda.md)

Supporting figure(s) live under [`figures/`](figures/).
