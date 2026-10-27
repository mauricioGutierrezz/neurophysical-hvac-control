# SPDX-License-Identifier: Apache-2.0
"""D2-BASE HVAC policy architecture; trained parameters are not included.

This module contains the BASE arm only. It uses public building observations
and metadata, maintains causal state, and projects neural proposals into the
official action shape. A trained parameter set must be supplied explicitly.
"""
import re
import numpy as np

ARM = 'BASE'
ZONE_FEATURES = 17
AHU_FEATURES = 12
HIDDEN = 32


def initial_weights(seed=20261020):
    rng = np.random.default_rng(seed)
    shapes = {'ze': (ZONE_FEATURES, HIDDEN),
              'ae': (2*HIDDEN+1+AHU_FEATURES, HIDDEN),
              'zh': (2*HIDDEN+ZONE_FEATURES, 2),
              'ah': (HIDDEN+AHU_FEATURES, 4),
              'we': (2*HIDDEN+1+AHU_FEATURES+3, 16), 'wh': (16, 3)}
    weights = {}
    for key, (n, m) in shapes.items():
        weights[key+'_w'] = rng.normal(0, np.sqrt(2/(n+m)), (n,m))
        weights[key+'_b'] = np.zeros(m)
    return weights


def pool(values, groups):
    """Batched invariant mean/max/count; empty groups map to zero."""
    b, _, h = values.shape
    result = np.zeros((b, len(groups), 2*h+1))
    for a, ids in enumerate(groups):
        if len(ids):
            result[:,a,:h] = values[:,ids].mean(axis=1)
            result[:,a,h:2*h] = values[:,ids].max(axis=1)
        result[:,a,-1] = np.log1p(len(ids))/5
    return result


def forward(weights, zone, ahu, water_previous, groups, zone_ahu):
    """One shared graph for both batch training and single-step inference."""
    p = weights
    z = np.tanh(zone @ p['ze_w'] + p['ze_b'])
    ai = np.concatenate([pool(z, groups), ahu], axis=-1)
    a = np.tanh(ai @ p['ae_w'] + p['ae_b'])
    hi = np.concatenate([a, ahu], axis=-1)
    wi = np.concatenate([pool(a, [np.arange(a.shape[1])])[:,0], ahu.mean(axis=1), water_previous], axis=-1)
    w = np.tanh(wi @ p['we_w'] + p['we_b'])
    ao = hi @ p['ah_w'] + p['ah_b']
    wo = w @ p['wh_w'] + p['wh_b']
    ao = np.concatenate([np.tanh(ao[...,:3]),ao[...,3:]],axis=-1)
    wo = np.concatenate([np.tanh(wo[...,:2]),wo[...,2:]],axis=-1)
    zi = np.concatenate([z, a[:,zone_ahu], zone], axis=-1)
    zo = np.tanh(zi @ p['zh_w'] + p['zh_b'])
    cache = (zone, ahu, z, ai, a, zi, hi, wi, w, ao, wo)
    return (zo, ao, wo), cache


class D2Policy:
    name = 'd2'
    arm = ARM

    def __init__(self, building_meta_data, action_spec):
        m = building_meta_data
        self.meta = m
        self.fields = list(m['observation_field_names'])
        self.index = {f:i for i,f in enumerate(self.fields)}
        self.nc, self.nd = m['n_continuous_action'], m['n_discrete_action']
        if action_spec['continuous_action'].shape != (self.nc,) or action_spec['discrete_action'].shape != (self.nd,):
            raise ValueError('Metadata/action-spec shape mismatch')
        self.ahu_ids = sorted(m['ahu_zone_assignments'], key=lambda name:int(re.search(r'\d+',name).group()))
        self.na = len(self.ahu_ids)
        if self.na < 1 or len(m['temp_indices']) != self.na or self.nd != self.na+1:
            raise ValueError('Unsupported official AHU/action mapping')
        self.zone_ids = sorted(m['zone_action_indices'])
        self.nz = len(self.zone_ids)
        if not self.nz:
            raise ValueError('At least one zone is required')
        assignments = [(room.replace('room_', 'VAV_'),a) for a,aid in enumerate(self.ahu_ids)
                       for room in m['ahu_zone_assignments'][aid]]
        zone_map = dict(assignments)
        if len(assignments)!=len(zone_map) or set(zone_map)!=set(self.zone_ids):
            raise ValueError('Each controlled zone must belong to exactly one AHU')
        self.zone_ahu = np.array([zone_map[z] for z in self.zone_ids], dtype=int)
        self.groups = [np.flatnonzero(self.zone_ahu==a) for a in range(self.na)]
        self.temp_obs = np.array([self.index[z+'_zone_air_temperature_sensor'] for z in self.zone_ids])
        self.zone_actions = np.array([[m['zone_action_indices'][z][name] for name in
            ('supply_air_damper_percentage_command','reheat_valve_percentage_command')] for z in self.zone_ids])
        self.ahu_actions = np.array(list(zip(m['temp_indices'],m['pressure_indices'][:self.na],m['recirculation_indices'])))
        self.water_actions = np.array([m['water_indices'][0],m['pressure_indices'][self.na]])
        owned = np.r_[self.zone_actions.ravel(),self.ahu_actions.ravel(),self.water_actions]
        if len(owned)!=self.nc or set(owned)!=set(range(self.nc)):
            raise ValueError('Incomplete or duplicated official actuator mapping')
        normal = m['observation_normalization_metadata']['zone_air_temperature_sensor']
        self.tmean, self.tstd = normal['sample_mean'], normal['sample_std']
        self.outside_obs = [self.index.get('ahs_'+a+'_outside_air_temperature_sensor') for a in self.ahu_ids]
        self.clock_obs = [self.index[n] for n in ('hod_sin_000','hod_cos_000','dow_sin_000','dow_cos_000')]
        self.parameters = None
        self.reset()

    def set_weights(self, weights):
        expected = initial_weights()
        if set(weights)!=set(expected) or any(np.shape(weights[k])!=expected[k].shape or not np.isfinite(weights[k]).all() for k in expected):
            raise ValueError('Invalid D2 weights')
        self.parameters = {k:np.asarray(v,dtype=np.float64).copy() for k,v in weights.items()}

    def reset(self):
        self.previous_temperature = None
        self.zone_integral = np.zeros(self.nz)
        self.ahu_integral = np.zeros(self.na)
        self.previous_continuous = np.full(self.nc,-1.,dtype=np.float32)
        self.previous_continuous[self.ahu_actions[:,2]] = 1.
        self.previous_discrete = np.zeros(self.nd,dtype=np.int32)

    def features(self, obs):
        """Advance causal temperature/integral state once, before the decision.

        R-D2-01: comfort_mode_now is authoritative; soon remains separate. When
        now is absent, use the causal P1-1a five-minute clock fallback.
        """
        obs = np.asarray(obs)
        if obs.shape!=(len(self.fields),) or not np.isfinite(obs).all():
            raise ValueError('Invalid public observation')
        # Float64 before denormalization avoids loss of precision in feature math.
        temperature = obs[self.temp_obs].astype(np.float64)*self.tstd+self.tmean
        clock = obs[self.clock_obs].astype(np.float64)
        if 'comfort_mode_now' in self.index:
            comfort = bool(obs[self.index['comfort_mode_now']] > .5)
        else:
            hour_phase=24*(np.arctan2(clock[0],clock[1])%(2*np.pi))/(2*np.pi)
            minute_slot=int(np.floor((hour_phase*60+1e-3)/5))%(24*12)
            day_phase=7*(np.arctan2(clock[2],clock[3])%(2*np.pi))/(2*np.pi)
            day=int(np.floor(day_phase+.5))%7
            hour=minute_slot//12
            comfort=day<5 and self.meta['morning_start_hour']<=hour<self.meta['evening_start_hour']
        low, high = self.meta['comfort_temp_window' if comfort else 'eco_temp_window']
        error = np.maximum(low-temperature,0)-np.maximum(temperature-high,0)
        delta = np.zeros(self.nz) if self.previous_temperature is None else temperature-self.previous_temperature
        self.zone_integral = np.clip(np.where(error==0, self.zone_integral*0.5, self.zone_integral+error),-10,10)
        self.previous_temperature = temperature.copy()
        outside = np.array([float(obs[i]) if i is not None else 285. for i in self.outside_obs])
        now = float(obs[self.index['comfort_mode_now']]) if 'comfort_mode_now' in self.index else float(comfort)
        soon = float(obs[self.index['comfort_mode_soon']]) if 'comfort_mode_soon' in self.index else 0.
        zone = np.column_stack([(temperature-295)/10,(low-temperature)/5,(temperature-high)/5,error/5,
            np.clip(delta,-2,2)/2,self.zone_integral/10,self.previous_continuous[self.zone_actions],
            (outside[self.zone_ahu]-285)/20,np.full(self.nz,now),np.full(self.nz,soon),
            np.tile(clock,(self.nz,1)),np.full(self.nz,(low-295)/10),np.full(self.nz,(high-295)/10)])
        ahu = np.zeros((self.na,AHU_FEATURES))
        for a, ids in enumerate(self.groups):
            if not len(ids): continue
            mean_temperature = temperature[ids].mean()
            mean_error = max(low-mean_temperature,0)-max(mean_temperature-high,0)
            self.ahu_integral[a] = np.clip(self.ahu_integral[a]*0.5 if mean_error==0 else self.ahu_integral[a]+mean_error,-10,10)
            ahu[a] = [(mean_temperature-295)/10,mean_error/5,error[ids].mean()/5,
                max(error[ids].max(),0)/5,max(-error[ids].min(),0)/5,np.clip(delta[ids].mean(),-2,2)/2,
                self.ahu_integral[a]/10,*self.previous_continuous[self.ahu_actions[a]],
                self.previous_discrete[a],(outside[a]-285)/20]
        water_previous = np.r_[self.previous_continuous[self.water_actions],self.previous_discrete[-1]]
        return zone, ahu, water_previous, error

    def remember(self, action):
        """Store the executed action (teacher during training, network online)."""
        self.previous_continuous = action['continuous_action'].copy()
        self.previous_discrete = action['discrete_action'].copy()

    def project(self, outputs, error):
        """Hard rest/off projection; never a PID fallback or teacher invocation."""
        zo, ao, wo = outputs
        continuous = np.full(self.nc,-1.,dtype=np.float32)
        discrete = np.zeros(self.nd,dtype=np.int32)
        continuous[self.zone_actions] = np.clip(zo,-1,1)
        continuous[self.zone_actions[error==0,0]] = -1.
        continuous[self.zone_actions[error<=0,1]] = -1.
        for a, ids in enumerate(self.groups):
            active = len(ids)>0 and np.any(error[ids]!=0) and ao[a,3]>=0
            if active:
                discrete[a] = 1
                continuous[self.ahu_actions[a]] = np.clip(ao[a,:3],-1,1)
            else:
                continuous[self.ahu_actions[a,2]] = 1.
        if np.any(discrete[:-1]) and wo[2]>=0:
            discrete[-1] = 1
            continuous[self.water_actions] = np.clip(wo[:2],-1,1)
            if not np.any(error>0): continuous[self.water_actions[0]] = -1.
        return {'continuous_action':continuous,'discrete_action':discrete}

    def get_action(self, obs):
        if self.parameters is None:
            raise RuntimeError('D2-BASE requires explicit model weights; random control is disabled')
        z,a,w,error = self.features(obs)
        outputs,_ = forward(self.parameters,z[None],a[None],w[None],self.groups,self.zone_ahu)
        action = self.project(tuple(o[0] for o in outputs),error)
        self.remember(action)
        return action
