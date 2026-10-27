# Progress and lessons

[Back to the project](../README.md)

I’m building this architecture in pieces. This page separates implementation, local evaluation and research proposals. It intentionally does not publish experiment outcomes or numerical results.

## A working foundation

The current neural baseline shares representations across zones and air-handling units. It includes physical features, simple causal history, hierarchical outputs and action constraints. Those components are implemented and have been evaluated in local closed-loop runs.

This is a foundation for the five-block design, rather than a completed demonstration of the entire architecture. Handling different input and output sizes is not the same as controlling an unfamiliar building well.

## What is still being explored

| Direction | Where it stands |
|---|---|
| D2-BASE shared encoders, causal history, policy heads and action projection | Implemented; locally evaluated; source included here without trained weights |
| Additional context and action-message coordination | Implemented and locally evaluated in other private variants; not included here |
| Learning directly from control rewards | PPO mechanism implemented; technical preflight only, without real training collection |
| Richer recurrent memory | Proposed |
| Thermal relationships between neighboring zones | Proposed |
| Online thermal adaptation and specialized experts | Research ideas requiring separate justification |

“Implemented” means code exists; “evaluated locally” means relevant local closed-loop records exist; “proposed” means a research idea. This page reports those status distinctions without disclosing outcomes. Technical checks establish whether a mechanism behaves as intended, not whether it learns a better policy.

## What would count as progress

An added block should improve control beyond repeatedly used development cases, preserve comfort and justify its computational cost. Complete trajectories, appropriate comparators and controlled ablations matter more than a larger collection of modules.

This release publishes no experiment outcomes, scores, comparisons, or claims of hidden-test generalization. It is not a public replication package.

## The public code

The D2-BASE implementation is provided without trained weights. The synthetic
example and tests exercise software structure only. They do not add evidence of
trained control performance or complete the five-block research agenda. See the
[code walkthrough](code.md).
