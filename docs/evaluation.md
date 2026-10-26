# Evaluation principles

[Overview](../README.md) · [Research status](research-status.md)

These are criteria for interpreting and planning research. They are not a claim that every stage below has been completed.

## Ask a controlled question

Compare a proposed mechanism with the existing baseline and an appropriate simpler alternative. When adding memory, distinguish temporal information from extra parameter capacity. When adding thermal connectivity, distinguish useful physical relationships from arbitrary graph structure. Improvement in an offline loss is insufficient to promote a controller.

## Preserve the information boundary

At action time, use only permitted observations, metadata and causal history. Keep privileged evaluator signals outside inference. Fit learned preprocessing on training data only, respect episode boundaries and reset temporal state. Future outcomes may be learning targets; they must not become inputs available before those outcomes occur.

## Evaluate complete trajectories

Smoke checks establish that a policy runs. Complete trajectories test closed-loop behavior. A single complete development episode is still not the full benchmark. Reused development cases must not be relabeled as blind validation; local reserved evaluation and the competition's hidden test are also distinct.

## Inspect trade-offs and failures

Track the official objective together with its comfort, cost and carbon components. Examine severe occupied excursions and their duration, not only averages. Command positions and equipment run time are diagnostics, not direct substitutes for energy or heat measurements. Report negative findings and limitations alongside improvements.

## Preserve provenance

Record the environment version, episode horizon, policy version, seeds and comparator provenance internally. Reuse a baseline only when its conditions are compatible. Measure inference cost separately from simulator time, and include the cost of model selection. Repeated deterministic trajectories do not provide independent uncertainty estimates.

## Scope of this public artifact

This repository documents the architecture and interpretation of evidence. It intentionally omits datasets, exact split assignments, tuning settings, trained weights, controller code and internal experiment artifacts. Its qualitative status statements are not a public replication package.
