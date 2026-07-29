# Research Memos

Individual research memoranda for PSYCH 755, following the course template
(question → results summary → remaining uncertainties).

| Memo | Researcher | Question |
|---|---|---|
| [`feature_predictive_power_ml_llm.md`](feature_predictive_power_ml_llm.md) | Jack J. Burleson | Which features have the greatest predictive power for CA under ML and LLM agents (SHAP + F1)? |
| [`transit_riders_ca.md`](transit_riders_ca.md) | Jack J. Burleson | Do regular public-transit riders differ in CA from non-regular riders? |
| [`ca_scores_predict_transit.md`](ca_scores_predict_transit.md) | Jack J. Burleson | Do group & interpersonal CA scores predict regular public-transit use? |
| [`geo_predicts_transit.md`](geo_predicts_transit.md) | Jack J. Burleson | Does geographical location predict regular public transit use? |
| [`car_access_predicts_transit.md`](car_access_predicts_transit.md) | Jack J. Burleson | Do car license & access (`Q20`/`Q21`) predict regular transit? |
| [`employment_predicts_transit.md`](employment_predicts_transit.md) | Jack J. Burleson | Does employment status predict regular transit? |
| [`rideshare_predicts_transit.md`](rideshare_predicts_transit.md) | Jack J. Burleson | Does ride-share frequency (`Q28`/`Q29`) predict regular transit? |
| [`q27_q28_predict_transit.md`](q27_q28_predict_transit.md) | Jack J. Burleson | Do Q27 (transit intensity) & Q28 (ride-share days) predict regular transit in traditional ML? |
| [`transit_covariate_followups.md`](transit_covariate_followups.md) | Jack J. Burleson | Head-to-head geo-memo follow-ups (car / employment / ride-share) |
| [`demographics_predict_transit.md`](demographics_predict_transit.md) | Jack J. Burleson | Do Age, Sex, and Student status predict regular transit? |
| [`country_predicts_transit.md`](country_predicts_transit.md) | Jack J. Burleson | Does country of residence predict regular transit (vs lat/long)? |
| [`q28_conditioned_on_car.md`](q28_conditioned_on_car.md) | Jack J. Burleson | Does Q28 retain lift after conditioning on car access? |
| [`ca_mobility_joint_predicts_transit.md`](ca_mobility_joint_predicts_transit.md) | Jack J. Burleson | Do CA scores add to Q28 and car access on a complete-case frame? |
| [`country_car_predicts_transit.md`](country_car_predicts_transit.md) | Jack J. Burleson | Do country and car access jointly predict regular transit? |
| [`q27_intensity_among_riders.md`](q27_intensity_among_riders.md) | Jack J. Burleson | What predicts Q27 intensity among already-regular riders? |
| [`common_n_head_to_head.md`](common_n_head_to_head.md) | Jack J. Burleson | Equal complete-case ranking of major transit predictors |
| [`residual_ca_after_rideshare.md`](residual_ca_after_rideshare.md) | Jack J. Burleson | Does CA still separate riders after accounting for Q28? |
| [`mi_head_to_head.md`](mi_head_to_head.md) | Jack J. Burleson | Does MI restore demos/CA or attenuate Q28’s lead? |
| [`transit_focus_regular_and_intensity.md`](transit_focus_regular_and_intensity.md) | Jack J. Burleson | TF1/TF2: predict regular transit & intensity with mobility held out? |
| [`vllm_v1_cross_model_comparison.md`](vllm_v1_cross_model_comparison.md) | Jack J. Burleson | Do v1 vLLM models recover PRCA as digital twins (cross-model)? |
| [`vllm_v1_llama31_8b.md`](vllm_v1_llama31_8b.md) | Jack J. Burleson | Does Llama-3.1-8B recover PRCA on prompt v1? |
| [`vllm_v1_llama32_3b.md`](vllm_v1_llama32_3b.md) | Jack J. Burleson | Does Llama-3.2-3B-Instruct recover PRCA on prompt v1? |
| [`vllm_v1_deepseek_r1_distill.md`](vllm_v1_deepseek_r1_distill.md) | Jack J. Burleson | Does DeepSeek-R1-Distill-Llama-8B recover PRCA on prompt v1? |
| [`vllm_v1_llama33_70b.md`](vllm_v1_llama33_70b.md) | Jack J. Burleson | Does Llama-3.3-70B recover PRCA on prompt v1? (mode collapse) |

Agenda / hub write-up: [`docs/research_memo_agenda.md`](../docs/research_memo_agenda.md) · [`docs/secondary_rq_followup_experiments.md`](../docs/secondary_rq_followup_experiments.md) · LLM wave: [`docs/llm_vllm_memo_agenda.md`](../docs/llm_vllm_memo_agenda.md)

Supporting figure(s) live under [`figures/`](figures/).
