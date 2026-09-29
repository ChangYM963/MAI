# Paper aggregates

These CSV files transcribe rounded values from the supplied final English manuscript, `ANOR_final.tex`. They contain **paper results**, not generated demo observations. They are aggregate extracts, not raw data or an independent reanalysis. Units appear in column names; fractions and percentages must not be interchanged.

| File | Manuscript table label | Content |
|:--|:--|:--|
| [initial_validation.csv](initial_validation.csv) | `tab:herding_validation` | Seven models, strict epsilon=0, 50 tasks each |
| [external_interventions.csv](external_interventions.csv) | `tab:external_control_reported` | Prompt/post-processing conditional drift and positive proportions |
| [internal_interventions.csv](internal_interventions.csv) | `tab:internal_control_reported` (upper panel) | Nine model/layer/trigger settings, shared-baseline comparison |
| [reliability_accuracy.csv](reliability_accuracy.csv) | `tab:reliability_conditions` | All A–D accuracy estimates and 95% intervals |
| [reliability_effects.csv](reliability_effects.csv) | `tab:reliability_primary_effects` | Primary contrasts and both Holm correction families |

Reported changes are copied as printed, so subtracting the rounded displayed endpoints can differ in the last decimal. External prompt baselines are model-dependent; inspect `baseline_type` before comparing. All prompt comparisons use the 30 tasks excluded from calibration; Mistral has 29 valid tasks. Internal shared-baseline values must not be interpreted as within-run pre/post inference.

Confidence intervals require the original paired observations to recompute. No missing raw observations or unavailable Doubao uncertainty estimates have been fabricated. Source hashes and table identifiers are recorded in [provenance.json](provenance.json); the figure index is [here](../../figures/README.md).
