# Method and implementation

[Overview](../README.md) · [Usage](usage.md) · [Reproduction scope](reproducibility.md)

## Task construction

`construction.build_tasks` implements five roles: screening, normalization, optional rewriting, option rendering, and quality checking. Structured mode works on supplied fields; `construct --semantic` asks a configured chat model to perform each role and records its raw JSON response.

A source needs a unique `id`, `market_context`, `instrument`, `decision_target`, `decision_axis`, explicit nonempty `constraints`, and optional `task_prompt` and `assumptions`. The reference renderer supports the `position_adjustment` axis: strongly increase, moderately increase, hold, moderately reduce, strongly reduce. This restricted schema makes the example inspectable. Other axes require a corresponding renderer and validator.

Only accepted tasks enter the pipeline. Added assumptions force review through the flagged route; missing fields, unusable JSON, unsupported axes, and failed quality checks are rejected with reasons. Semantic output needs factual review before use in a research dataset: structural validation cannot prove that a model preserved every source fact. Contextual tasks have no answer key.

## Contextual distributions and fixed references

For each agent and stage, normalize counts of valid A–E samples and mix them with the same assigned prior:

```math
p=(1-\beta)f+\beta q,\qquad \beta=0.35.
```

The included agent profiles and priors are illustrative. Unlike the reliability protocol, contextual estimation can use the remaining valid responses when some samples are invalid. An agent with no valid sample invalidates that task repetition.

The maximum of the mean initial distribution determines the reference majority; a fixed A–E order resolves equal maxima. An agent belongs to the initial minority when its own initial argmax differs from that majority. Both quantities are frozen for all later branches. An empty minority makes minority drift undefined.

The MAI weights are 4 times one minus normalized ordinal distance to the majority. The score lies in [0,4]. For each task, average ΔMAI over initial-minority agents, then average repetitions before computing task-level summaries:

| Statistic | Definition |
|:--|:--|
| D | Mean signed task drift |
| H+ | Mean of max(task drift, 0), including zero/negative tasks in the denominator |
| Positive proportion | Fraction of tasks with drift above the chosen zero threshold |
| Conditional positive mean | Mean drift among positive tasks; undefined when none are positive |
| Switch rate | Fraction of initial-minority agents whose later argmax equals the initial majority |

General summaries use tolerance 1e-12; prompt-strength calibration uses the paper's strict threshold 0. A branch's descriptive summary excludes a task if any of its repetitions is invalid for that branch. Paired intervention intervals use tasks valid in every compared branch and repetition.

## Prompt calibration and post-processing

Split accepted tasks before calibration. For each calibration round, collect an initial response set and reuse it across the six candidate strengths. Compute each round's conditional mean positive drift, then average those round means. Select the lowest score, resolving ties toward the smaller strength. A candidate with an undefined round mean is not eligible; if none are eligible, record the reason and skip the selected-prompt branch.

On held-out tasks, evaluate ordinary social exposure, a zero-strength structured prompt, and the selected strength. Zero strength still contains structured instructions. Consequently, improvement over ordinary group information and incremental improvement over zero strength answer different questions. The report stores both comparisons.

Probability correction uses `p0 + (1 − lambda) * (p1 − p0)` on the same recorded distributions. The fixed MAI reference makes its drift equal to `(1 − lambda)` times the original drift. This is an algebraic property, not an independently learned intervention.

## Hidden-state intervention

On calibration observations from initial-minority agents:

1. Capture the final prompt-position hidden state at the chosen decoder block before/after group exposure.
2. Label an observation positive when the probability assigned to the frozen majority increases.
3. Normalize the mean hidden-state difference among positive observations to obtain direction `v`.
4. Fit a standardized logistic risk head on after-exposure hidden states. The paper's original model-specific head is an MLP; this small reference uses logistic regression for portability.
5. On held-out initial-minority observations with risk `s >= threshold`, compute the update below. If calibration has no usable positive direction, disable the controller with an explicit reason.

```math
u=-\alpha s\max(0,h^\mathsf{T}v)v.
```

The local adapter adds the same update at the final token position of the selected block on every forward pass during regeneration. It removes hooks in a `finally` block. Untriggered agents keep the original social responses. Triggered regeneration reuses the baseline sample seeds. The audit reports trigger status, risk, projection, and whether the empirical distribution actually changed. A trigger is not synonymous with a behavioral change.

The simulator uses synthetic hidden coordinates and a stipulated social-mixture response function. Its controller output is only a software demonstration. The local adapter accesses actual model tensors; availability of the adapter does not imply replication of the original checkpoints or settings.

## Exactly solvable reliability tasks

The options use allocations A–E = 1, 3/4, 1/2, 1/4, 0. The generator balances unique correct positions within each family:

| Family | Objective | Constraint |
|:--|:--|:--|
| Quadratic utility | Maximize `a*x − b*x*x` | Choose one of the five allocations |
| Loss budget | Maximize `gain*x` | `loss*x <= budget` |
| Liquidity budget | Maximize `gain*x` | `capital*(1 − x) >= reserve` |

Rational arithmetic determines the unique optimum. An independent oracle evaluates equivalent inequalities/objectives using integer quarter-units. Generated ties and duplicate parameter cases are rejected. `variants=4` gives 60 newly generated tasks, not the original paper dataset.

Each repetition obtains initial responses once and independently branches into A–D in randomized order. A and C still require a second answer. B and D receive an external panel of four identical wrong votes plus one correct vote, without correctness labels. The panel is independent of the agents' initial majority. Prompts contain the task facts and prior selected answer, never the answer key or solver explanation.

## Parsing, abstention, and denominators

- Strip surrounding whitespace and accept exactly one uppercase A, B, C, D, or E. Explanations, lowercase, empty strings, and multiple letters are invalid.
- Any invalid sample or a tied sample mode makes the agent abstain. Never silently repair a malformed answer.
- The system needs more than half of **all scheduled agents** to select one option. Abstentions remain in the denominator.
- Incorrect consensus needs at least `ceil(0.8 * N)` scheduled agents to select the same wrong option.
- Retention is pooled retained-correct count divided by pooled initially correct count, not an unweighted average of task ratios. A zero denominator is undefined.
- Reliability MAI uses empirical distributions without priors. Its common-valid subset requires valid samples for every agent, stage, A–D branch, and repetition, plus a nonempty initial minority. Invalid tasks remain in accuracy/abstention statistics.

## Paired inference

Repeated outcomes are first grouped by task. Bootstrap resampling is stratified by task family; every selected task carries all conditions and repetitions together. Retention is recomputed from counts on each resample. Percentile intervals use the 2.5th and 97.5th percentiles. Undefined ratio replicates are excluded and their count is reported.

Effects are accuracy loss `A − B`, direct recovery `D − B`, general verification `C − A`, and specific protection `(D − B) − (C − A)`. Two-sided task sign-flip tests remove zero differences, enumerate all signs for at most 16 nonzero tasks, and otherwise use Monte Carlo signs with the plus-one correction. These tests assume sign exchangeability under the null.

The pipeline applies Holm correction to the two primary tests (loss and specific protection) **within one run**. The paper also corrects all six primary tests jointly across its three models; do not treat a within-run corrected value as that six-test result. Contextual intervals are exploratory paired scenario intervals; full template-level robustness analyses are not implemented in this release.
