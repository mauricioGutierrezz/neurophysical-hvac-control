# SPDX-License-Identifier: Apache-2.0
"""Run a structural example: python -m neurophysical_hvac."""

import numpy as np

from . import HierarchicalEncoder


def main():
    print("Synthetic structure demo — untrained parameters, no control simulation")
    print("The same encoder instance processes two invented building layouts.\n")
    model = HierarchicalEncoder.random(4, 3, 2, seed=7)
    rng = np.random.default_rng(11)
    for name, mapping in [("Small office", [0, 0, 1, 1, 1]),
                          ("Larger office", [0, 1, 1, 2, 2, 3, 3, 3, 0])]:
        mapping = np.array(mapping)
        zones, ahus = len(mapping), int(mapping.max()) + 1
        # These are arbitrary prepared features, not temperatures or observations.
        encoded = model.encode(rng.normal(size=(zones, 4)),
                               rng.normal(size=(ahus, 3)), np.zeros(2), mapping)
        print(f"{name}: {zones} zones / {ahus} AHUs")
        print(f"  Zone embeddings: {encoded.zones.shape}")
        print(f"  AHU embeddings:  {encoded.ahus.shape}")
        print(f"  Water context:   {encoded.water.shape}")
    print("\nVariable shapes are supported. Transfer performance is not measured here.")


if __name__ == "__main__":
    main()
