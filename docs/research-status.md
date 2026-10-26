# Progress and lessons

[Back to the project](../README.md)

I’m building this architecture in pieces, using local experiments to decide what is worth keeping. This page describes the evidence reviewed on 26 October 2026.

## A working foundation

The current neural baseline shares representations across zones and air-handling units. It includes physical features, simple causal history, hierarchical outputs and action constraints. Those components are implemented and have been evaluated in local closed-loop runs.

This is a foundation for the five-block design, rather than a completed demonstration of the entire architecture. Handling different input and output sizes is not the same as controlling an unfamiliar building well.

## What the experiments taught me

**Representation matters, but local gains have limits.** A revised representation improved on an earlier neural reference in one known development case. It did not beat the conventional comparator there, and that experiment alone says little about transfer.

**More communication does not automatically help.** Variants that added context and same-step action messages were evaluated but not retained. The communication variant also worsened comfort relative to the neural baseline in that comparison.

**Imitation and control answer different questions.** Earlier transfer tests produced inconsistent benefits across the examined development cases. Matching a teacher more closely is not enough to establish better closed-loop behavior.

## What is still being explored

| Direction | Where it stands |
|---|---|
| Shared encoders and simple causal history | Implemented and locally evaluated |
| Additional context and action-message coordination | Evaluated locally; not retained |
| Learning directly from control rewards | PPO mechanism implemented; technical preflight only, without real training collection or demonstrated control improvement |
| Richer recurrent memory | Proposed |
| Thermal relationships between neighboring zones | Proposed |
| Online thermal adaptation and specialized experts | Research ideas requiring separate justification |

“Implemented” means code exists; “evaluated locally” means relevant local closed-loop records exist; “proposed” means a research idea. Technical checks establish whether a mechanism behaves as intended, not whether it learns a better policy.

## What would count as progress

An added block should improve control beyond repeatedly used development cases, preserve comfort and justify its computational cost. Complete trajectories, appropriate comparators and controlled ablations matter more than a larger collection of modules.

The reviewed local runs do not establish hidden-test generalization, a leaderboard result or real-building readiness. This public repository explains the design and lessons qualitatively; the private experiment artifacts are not a public replication package.
