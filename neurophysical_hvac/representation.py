# SPDX-License-Identifier: Apache-2.0
"""Zone -> air handler -> water context, without control or trained weights.

Extracted and adapted from the research implementation. The shared affine/tanh
encoders and mean/max/log-count aggregation retain the original computation.
This interface handles one building at a time and accepts prepared features.
"""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_array(values, ndim, name):
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite {ndim}-dimensional array")
    return array


def _count_scale(value):
    if not np.isscalar(value) or np.iscomplexobj(value):
        raise ValueError("count_scale must be finite and positive")
    scale = float(value)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("count_scale must be finite and positive")
    return scale


def pool_by_ahu(values, zone_to_ahu, n_ahus, *, count_scale=1.0):
    """Return [mean, max, log1p(count)/scale] for each equipment group.

    ``values`` has shape (zones, channels). Each zone belongs to exactly one
    AHU through an integer index in ``zone_to_ahu``. Empty groups return zeros.
    The mean and max do not depend on the order of zones within a group.
    """
    values = _finite_array(values, 2, "values")
    n_ahus = _positive_integer(n_ahus, "n_ahus")
    scale = _count_scale(count_scale)
    mapping = np.asarray(zone_to_ahu)
    if values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("values must contain at least one zone and channel")
    if mapping.shape != (len(values),) or mapping.dtype.kind not in "iu":
        raise ValueError("zone_to_ahu must contain one integer per zone")
    if np.any(mapping < 0) or np.any(mapping >= n_ahus):
        raise ValueError("zone_to_ahu contains an out-of-range AHU index")
    width = values.shape[1]
    pooled = np.zeros((n_ahus, 2 * width + 1))
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for group in range(n_ahus):
                members = values[mapping == group]
                if len(members):
                    pooled[group, :width] = members.mean(axis=0)
                    pooled[group, width:2 * width] = members.max(axis=0)
                pooled[group, -1] = np.log1p(len(members)) / scale
    except FloatingPointError as error:
        raise ValueError("aggregation exceeded the numerical range; check feature scaling") from error
    return pooled


@dataclass(frozen=True)
class DenseEncoder:
    """One affine/tanh map; the same parameters process every entity row."""

    weight: np.ndarray
    bias: np.ndarray

    def __post_init__(self):
        weight = _finite_array(self.weight, 2, "weight").copy()
        bias = _finite_array(self.bias, 1, "bias").copy()
        if min(weight.shape) < 1 or bias.shape != (weight.shape[1],):
            raise ValueError("weight and bias dimensions do not match")
        weight.flags.writeable = False
        bias.flags.writeable = False
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", bias)

    def __call__(self, features):
        features = _finite_array(features, 2, "features")
        if features.shape[1] != self.weight.shape[0]:
            raise ValueError("feature width does not match the encoder")
        try:
            with np.errstate(over="raise", invalid="raise"):
                return np.tanh(features @ self.weight + self.bias)
        except FloatingPointError as error:
            raise ValueError("encoder exceeded the numerical range; check feature scaling") from error


@dataclass(frozen=True)
class Embeddings:
    """Learned representation slots, not physical measurements or actions."""

    zones: np.ndarray
    ahus: np.ndarray
    water: np.ndarray


class HierarchicalEncoder:
    """Reuse three shared maps across variable zone and equipment counts."""

    def __init__(self, zone_encoder, ahu_encoder, water_encoder, *, count_scale=1.0):
        self.zone_encoder = zone_encoder
        self.ahu_encoder = ahu_encoder
        self.water_encoder = water_encoder
        self.count_scale = _count_scale(count_scale)
        width = zone_encoder.weight.shape[1]
        if ahu_encoder.weight.shape[1] != width:
            raise ValueError("zone and AHU embedding widths must match")
        self.ahu_features = ahu_encoder.weight.shape[0] - (2 * width + 1)
        self.water_features = water_encoder.weight.shape[0] - (2 * width + 1) - self.ahu_features
        _positive_integer(self.ahu_features, "AHU feature width")
        _positive_integer(self.water_features, "water feature width")

    @classmethod
    def random(cls, zone_features, ahu_features, water_features, *,
               width=8, water_width=6, seed=7, count_scale=1.0):
        """Create untrained, reproducible parameters for structural examples."""
        dims = [zone_features, ahu_features, water_features, width, water_width]
        for value in dims:
            _positive_integer(value, "feature and embedding dimensions")
        rng = np.random.default_rng(seed)

        def dense(inputs, outputs):
            weight = rng.normal(0, np.sqrt(2 / (inputs + outputs)), (inputs, outputs))
            return DenseEncoder(weight, np.zeros(outputs))

        return cls(
            dense(zone_features, width),
            dense(2 * width + 1 + ahu_features, width),
            dense(2 * width + 1 + ahu_features + water_features, water_width),
            count_scale=count_scale,
        )

    def encode(self, zone_features, ahu_features, water_previous, zone_to_ahu):
        """Encode prepared, current/past features for one building.

        The caller owns units, scaling and chronological construction of the
        inputs. This stateless method neither reads a simulator nor advances
        temporal state. ``water_previous`` is a prepared history vector.
        """
        zone_features = _finite_array(zone_features, 2, "zone_features")
        ahu_features = _finite_array(ahu_features, 2, "ahu_features")
        water_previous = _finite_array(water_previous, 1, "water_previous")
        if ahu_features.shape[1] != self.ahu_features or len(ahu_features) < 1:
            raise ValueError("invalid AHU feature shape")
        if water_previous.shape != (self.water_features,):
            raise ValueError("invalid water history shape")
        zones = self.zone_encoder(zone_features)
        zone_context = pool_by_ahu(zones, zone_to_ahu, len(ahu_features),
                                   count_scale=self.count_scale)
        ahus = self.ahu_encoder(np.concatenate([zone_context, ahu_features], axis=1))
        building_context = pool_by_ahu(ahus, np.zeros(len(ahus), dtype=int), 1,
                                       count_scale=self.count_scale)[0]
        try:
            with np.errstate(over="raise", invalid="raise"):
                mean_ahu_features = ahu_features.mean(axis=0)
        except FloatingPointError as error:
            raise ValueError("AHU mean exceeded the numerical range; check feature scaling") from error
        water_input = np.concatenate([building_context, mean_ahu_features, water_previous])
        water = self.water_encoder(water_input[None])[0]
        return Embeddings(zones, ahus, water)
