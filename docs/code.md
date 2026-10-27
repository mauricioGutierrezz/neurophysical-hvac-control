# Run the shared representation

[Back to the project](../README.md)

This repository includes the D2-BASE policy source and a shared representation
module. D2-BASE contains the feature/state construction, neural forward pass,
action heads and operational projection for one implemented policy version.
The example uses invented feature arrays and untrained random parameters. It
does not use a simulator or measure HVAC performance. No quantitative results
are published here.

## Start here

Tested with Python 3.12 and NumPy 2.0.2. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m neurophysical_hvac
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate` instead. Only NumPy is needed;
the tests use Python's standard library. No competition downloads or simulator
installation are required.

The demo processes two invented layouts through the same encoder instance. It
prints zone, AHU and water-context array shapes. Five zones and two AHUs become
`(5, 8)` zone embeddings and `(2, 8)` AHU embeddings. Nine zones and four AHUs
become `(9, 8)` and `(4, 8)`. Both produce a six-channel global water context.
Those numbers describe the synthetic example, not a trained model specification.

## Read the code

| File | Purpose |
|---|---|
| [representation.py](../neurophysical_hvac/representation.py) | Shared encoders and hierarchical aggregation |
| [d2_base.py](../neurophysical_hvac/d2_base.py) | D2-BASE causal state, policy heads and action projection |
| [__main__.py](../neurophysical_hvac/__main__.py) | Two synthetic layouts, one untrained encoder |
| [test_d2_base.py](../tests/test_d2_base.py) | Synthetic temporal state, policy outputs, action projection and weight contract |
| [test_representation.py](../tests/test_representation.py) | Numerical aggregation, permutation, shape and input contracts |

The flow is short:

1. A shared affine/tanh map encodes each zone's prepared features.
2. Zone embeddings are grouped by their AHU. Each group contributes its mean,
   maximum and scaled log-count alongside the AHU's local features.
3. Another shared affine/tanh map encodes each AHU.
4. AHU embeddings are aggregated into building context, then combined with
   mean AHU features and a prepared water-history vector for the water encoder.

Mean pooling summarizes collective conditions. Maximum pooling retains extreme
embedding values that an average can hide; these latent values are not measured
temperatures or proven indicators of the most uncomfortable zone. Log-count
records group size with configurable scaling. Empty AHU groups contribute a
zero zone summary while the AHU can still have its own local features.

## Input contract

`HierarchicalEncoder.encode` takes one building at a time:

| Argument | Shape | Meaning |
|---|---|---|
| `zone_features` | `(zones, zone_channels)` | Prepared numerical features for each zone |
| `ahu_features` | `(ahus, ahu_channels)` | Prepared numerical features for each air handler |
| `water_previous` | `(water_channels,)` | Prepared history vector for shared water context |
| `zone_to_ahu` | `(zones,)`, integers | Exactly one valid AHU index per zone |

At least one zone and AHU are required. Arrays must be finite, with feature
widths matching the encoders. Individual AHUs may have no zones. The caller owns
units, normalization and chronology; this module does not parse official sensor
metadata, maintain temporal state or check whether supplied features leak future
information. Its forward pass is stateless.

Reordering zones reorders their embeddings while preserving the AHU and global
results up to floating-point rounding. Reordering AHUs requires relabeling the
zone-to-AHU mapping; global context then remains invariant. These are structural
properties, not evidence of transfer performance.

## D2-BASE parameters and limits

The policy initializes without embedded parameters. `D2Policy.set_weights()`
accepts an explicit, shape-checked parameter mapping, but this repository does
not provide a trained mapping. A deterministic random initialization is only
appropriate for structural tests; its actions have no demonstrated control
value. Official observation metadata and action specifications are required to
construct the policy. The policy code alone is not a drop-in competition
submission or a reproducible trained controller.

## What is adapted, and what is omitted

The shared affine/tanh operations and mean/max/log-count hierarchy come from the
existing research code. This extraction gives them explicit names, validates
inputs, accepts one building per call and makes feature widths configurable.
Private numerical checks compare the extraction with its source computation
using synthetic features and untrained parameters. The public tests stand alone.

The release includes one policy version, D2-BASE. It excludes trained weights,
the training loop, experiment data and trajectories, reward/evaluation
implementation, and quantitative results. Other D2 variants are not included.
The remaining architecture blocks are explained in the [architecture
notes](architecture.md), with implementation status distinguished from
proposed research directions.
