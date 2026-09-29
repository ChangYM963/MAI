# Paper results and interpretation

[Overview](../README.md) · [Aggregate CSVs](../data/paper/README.md) · [All figures](../figures/README.md)

All results on this page are transcribed from the supplied final manuscript. Offline demo reports are separate software examples. The CSVs preserve the manuscript's displayed precision, so rounding can produce small differences when recomputing a change from printed endpoints.

## Initial validation

The study includes 50 contextual tasks per model, eleven agents, and ten samples per agent/stage. Five of the six models with estimable uncertainty have scenario-bootstrap intervals for mean signed drift entirely above zero. Llama's signed-drift scenario interval spans zero, although its conditional positive drift is relatively large. Doubao has only aggregate point estimates.

| Model | Signed drift D [95% scenario CI] | Positive burden H+ [95% scenario CI] |
|:--|:--|:--|
| Qwen Plus | 0.0820 [0.0588, 0.1056] | 0.0823 [0.0591, 0.1058] |
| DeepSeek-v3 | 0.0572 [0.0374, 0.0795] | 0.0572 [0.0374, 0.0795] |
| GLM-4.6 | 0.2069 [0.1744, 0.2408] | 0.2069 [0.1744, 0.2408] |
| Qwen2.5-7B | 0.1091 [0.0746, 0.1449] | 0.1098 [0.0757, 0.1452] |
| Mistral-7B | 0.0997 [0.0337, 0.1862] | 0.1010 [0.0353, 0.1870] |
| Llama-3.1-8B | 0.0449 [−0.0142, 0.1051] | 0.1028 [0.0619, 0.1480] |

Source: `tab:initial_drift_inference`. Counts in the overview use epsilon=0; introducing tolerance 1e-12 changes some prevalence estimates without changing reported D/H+ precision. Template-block sensitivity is analyzed separately in the paper.

## External control

Selected prompt strengths are 0.8, 1.0, 0, 0.2, 0, and 0.6 for Qwen Plus, DeepSeek-v3, GLM-4.6, Qwen2.5-7B, Mistral-7B, and Llama-3.1-8B. Selection uses 20 tasks, two rounds per level; prompt evaluation uses the remaining 30. Post-processing uses all 50 tasks. Mistral loses one task with an empty initial minority.

| Model | Prompt conditional positive drift, baseline → control | Positive proportion, baseline → control |
|:--|:--|:--|
| Qwen Plus | 0.1274 → 0.0083 | 63.33% → 30.00% |
| DeepSeek-v3 | 0.0851 → 0.0818 | 60.00% → 66.67% |
| GLM-4.6 | 0.0864 → 0.0864 | 60.00% → 60.00% |
| Qwen2.5-7B | 0.2617 → 0.3844 | 60.00% → 63.33% |
| Mistral-7B | 0.0584 → 0.0584 | 41.38% → 41.38% |
| Llama-3.1-8B | 0.2777 → 0.2900 | 100.00% → 93.33% |

Source: `tab:external_control_reported`. Qwen Plus and DeepSeek use shared baselines here; GLM and local models use zero-strength prompts. These are changes in **conditional positive drift**, not unconditional H+.

In the primary incremental H+ comparison, every selected prompt is compared against its zero-strength version. Qwen Plus's gain is −0.0004 [−0.0037, 0.0025]; its improvement relative to the shared baseline is therefore not evidence that increasing strength adds benefit. Qwen2.5-7B's H+ gain is about −0.087, with interval [−0.142, −0.037]. GLM and Mistral select zero and do not establish a nonzero-strength benefit. Post-processing at lambda=0.6 reduces fixed-reference drift algebraically; threshold-sensitive positive counts can still be affected by floating-point near-zero values.

## Internal control

| Model | Paper layer | (alpha, threshold) | Conditional positive drift, shared baseline → control |
|:--|--:|:--|:--|
| Qwen2.5-7B | 14 | (1.5, 0.1) | 0.1523 → 0.1474 |
| Qwen2.5-7B | 19 | (0.1, 0.8) | 0.1523 → 0.1423 |
| Qwen2.5-7B | 25 | (1.5, 0.7) | 0.1523 → 0.1014 |
| Mistral-7B | 14 | (0.2, 0.1) | 0.2084 → 0.1967 |
| Mistral-7B | 19 | (0.7, 0.5) | 0.2084 → 0.2331 |
| Mistral-7B | 25 | (1.5, 0.4) | 0.2084 → 0.2063 |
| Llama-3.1-8B | 14 | (0.1, 0.8) | 0.1725 → 0.2112 |
| Llama-3.1-8B | 19 | (0.4, 0.4) | 0.1725 → 0.1687 |
| Llama-3.1-8B | 25 | (1.0, 0.1) | 0.1725 → 0.1777 |

Source: `tab:internal_control_reported`. These shared-baseline results and the paper's within-run pre/post paired analyses use different baselines. Preserve that distinction. Trigger frequency and changed-output frequency are also different; see the paper's internal-intervention figure. Layer settings are reported as in the manuscript; the reference adapter explicitly uses zero-based block indices.

## Reliability

System accuracy (%) and task-bootstrap 95% intervals:

| Model | A | B | C | D |
|:--|:--|:--|:--|:--|
| Qwen2.5-7B | 21.67 [11.67, 33.33] | 0.00 [0.00, 0.00] | 22.50 [12.50, 33.33] | 2.50 [0.00, 6.67] |
| Mistral-7B | 19.17 [10.00, 29.17] | 10.83 [4.17, 18.33] | 22.50 [12.50, 33.33] | 13.33 [5.83, 21.67] |
| Llama-3.1-8B | 17.50 [8.33, 27.50] | 8.33 [2.50, 15.00] | 9.17 [4.17, 15.00] | 10.00 [4.17, 16.67] |

Source: `tab:reliability_conditions`. Each model has 60 tasks (20 per family), two repetitions, five agents, and three samples per stage. Qwen uses unconstrained generation; Mistral and Llama use constrained decoding. A system abstention counts as unsuccessful.

| Model | Contrast | Effect in pp [95% CI] | Within-model Holm p | Joint six-test Holm p |
|:--|:--|:--|--:|--:|
| Qwen2.5-7B | A − B | 21.67 [11.67, 33.33] | 0.000488 | 0.001465 |
| Qwen2.5-7B | (D − B) − (C − A) | 1.67 [−1.67, 5.83] | 0.750000 | 1.000000 |
| Mistral-7B | A − B | 8.33 [1.67, 15.83] | 0.078125 | 0.156250 |
| Mistral-7B | (D − B) − (C − A) | −0.83 [−8.33, 6.67] | 1.000000 | 1.000000 |
| Llama-3.1-8B | A − B | 9.17 [2.50, 16.67] | 0.046875 | 0.117188 |
| Llama-3.1-8B | (D − B) − (C − A) | 10.00 [0.83, 19.17] | 0.066293 | 0.198880 |

Source: `tab:reliability_primary_effects`. Qwen's accuracy loss passes both correction families. No specific-protection effect passes its specified Holm correction. Direct recovery under the wrong panel is only 1.67–2.50 percentage points.

Auxiliary outcomes explain why a lower wrong-consensus rate need not imply reliable decisions. For Llama, B → D reduces incorrect consensus from 58.33% to 15.83%, while correct-judgment retention falls from 53.33% to 24.44% and agent abstention rises from 7.17% to 28.17%. Accuracy rises only from 8.33% to 10.00%. Source: `tab:reliability_auxiliary`.

## Reading the figures

All nine original figure files appear in the overview and in the [figure index](../figures/README.md). Click an image to view full resolution. Figures retain their original statistical definitions and baselines; the public simulation does not regenerate these published plots.
