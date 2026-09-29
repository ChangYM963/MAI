# Paper-to-code coverage

The release makes the full experiment structure executable and exposes intermediate artifacts. It includes original paper figures and transcribed aggregates, but does not claim exact numerical reproduction of the published experiments.

| Component | Paper | Public reference |
|:--|:--|:--|
| Construction | Five functional roles with factual/constraint checks | Structured implementation plus optional model-based role calls and audit logs |
| Contextual material | 50 financial tasks across ten templates | 14 fictional source examples; 12 accepted |
| Priors/profiles | Model/agent-specific experimental assignments | Explicit illustrative profiles; same beta=0.35 mixing rule |
| Two stages | Initial distribution and frozen majority/minority | Implemented with saved raw answers and fixed references |
| Prompt calibration | 20 tasks, two rounds, six strengths; 30 held out | Same selection structure; demo uses 4/8 split and illustrative prompt wording |
| Post-processing | Lambda=0.6 probability correction | Implemented and checked against the exact drift-shrinkage identity |
| Hidden intervention | Model-specific MLP, layer/trigger settings | Portable logistic head and actual decoder-block hook; simulation also provided |
| Reliability tasks | 60 exactly solvable tasks/model | Balanced generator with two independent oracles; 15 per variant |
| A–D protocol | Shared initial responses, wrong panel, verification | Implemented with independent branches and randomized condition order |
| Output parsing | Any invalid sample/tie causes abstention | Implemented and tested, including empty/multi-letter responses |
| System vote | Strict majority of all scheduled agents | Implemented; abstainers stay in the denominator |
| Reliability MAI | Common-valid tasks across agents/stages/conditions/repetitions | Implemented with explicit eligible counts |
| Statistical inference | Paired task bootstrap, sign-flip tests, Holm correction | Implemented within one model run; no pooled six-test correction across models |
| Contextual robustness | Scenario and template-block sensitivity | Exploratory paired scenario intervals; template analysis not included |
| Decoding | Qwen unconstrained; Mistral/Llama constrained in reliability | Generic chat/local adapters use unconstrained sampling and strict parsing |
| Results | Real model observations | Paper aggregates/figures separate from labeled simulation outputs |

## Verification performed

The tests cover independent answer oracles, balance and duplicate exclusion, strict parsing, fixed-reference MAI, algebraic correction, shared initial answers, answer-key non-disclosure, abstention denominators, paired resampling, task-level exclusions, sign-flip/Holm calculations, a mocked chat request, and the complete offline pipeline. A PyTorch test checks that the steering hook changes only the final sequence position and preserves other outputs; it skips when PyTorch is unavailable.

No paid API calls or full downloaded-model experiments are required for CI. They were not used to validate this release. Consequently, successful offline tests validate the reference protocol and plumbing; they do not validate scientific claims about actual LLM behavior. Evaluate adapters with the exact model, tokenizer, generation implementation, and hardware intended for an experiment.

## Interpreting the paper

MAI measures proximity to a fixed majority on an ordered scale. It is not a probability of correctness, a calibrated confidence estimate, or a causal estimate of accuracy loss. Contextual and synthetic reliability experiments are distinct. Low baseline accuracy, preset wrong-majority exposure, and different decoding protocols limit generalization.

When citing numerical results, use [paper aggregates](../data/paper/README.md) and [result notes](results.md), not `examples/output` or a simulator report. The aggregate CSVs do not supply raw observations and cannot independently regenerate the paper's confidence intervals. Original task sets, raw model responses, exact checkpoints, and original hidden-head weights are not included in this reference release.
