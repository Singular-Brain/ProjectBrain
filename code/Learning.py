import numpy as np

class Learning(object):
    def __init__(self):
        self.reward_based = False

### Online Learning
class FSTDP_online(Learning):
    def __init__(self, dt,
                 interval_time = 0.01, # seconds
                 pre_post = 1,
                 post_pre = -1,
                 **kwargs):
        """
        Flat STDP
        """
        super().__init__()
        self.interval = interval_time / dt #timepoints
        self.pre_post = pre_post
        self.post_pre = post_pre

    def __call__(self, network, neuron):
        current_timestep = neuron.timestep
        for preSN in network.predecessors(neuron):
            if preSN.spike_timepoints:
                diff = current_timestep - preSN.spike_timepoints[-1]
                if diff < self.interval:
                    network.edges.get((preSN, neuron))['weight'] += np.array([self.pre_post]) 
        for postSN in network.successors(neuron):
            if postSN.spike_timepoints:
                diff = current_timestep - postSN.spike_timepoints[-1]
                if diff < self.interval:
                    network.edges.get((neuron, postSN))['weight'] += np.array([self.post_pre]) 


class RFSTDP_online(Learning):
    def __init__(self, dt,
                 interval_time = 0.01, # seconds
                 pre_post = 1,
                 reward_pre_post = 2,
                 post_pre = -1,
                 reward_post_pre = 1,
                 ):
        """
        Reward-modulated Flat STDP 
        """
        super().__init__()
        self.reward_based = True
        self.interval = interval_time / dt #timepoints
        self.pre_post = pre_post
        self.post_pre = post_pre
        self.reward_pre_post = reward_pre_post
        self.reward_post_pre = reward_post_pre
        self.reward = False

    def __call__(self, network, neuron):
        current_timestep = neuron.timestep
        for preSN in network.predecessors(neuron):
            diff = current_timestep - preSN.last_spike_timepoint
            if diff < self.interval:
                if self.reward:
                    network.edges.get((preSN, neuron))['weight'] +=\
                    np.array([self.reward_pre_post]) 
                else:
                    network.edges.get((preSN, neuron))['weight'] +=\
                    np.array([self.pre_post]) 

        for postSN in network.successors(neuron):
            diff = current_timestep - postSN.last_spike_timepoint
            if diff < self.interval:
                if self.reward:
                    network.edges.get((neuron, postSN))['weight'] +=\
                    np.array([self.reward_post_pre]) 
                else:
                    network.edges.get((neuron, postSN))['weight'] +=\
                    np.array([self.post_pre]) 

    def set_reward_rule(self, reward_rule):
        self.reward_rule = reward_rule

    def set_reward(self):
        self.reward = self.reward_rule()


### Offline learning
class RFSTDP(Learning):
    def __init__(self, NeuronGroup,
                 interval_time = 0.01, # seconds
                 pre_post = 1,
                 reward_pre_post = 2,
                 post_pre = -1,
                 reward_post_pre = 1,
                 ):
        """
        Reward-modulated Flat STDP 
        """
        super().__init__()
        self.group = NeuronGroup
        self.network = NeuronGroup.network
        self.dt = NeuronGroup.dt
        self.reward_based = True
        self.interval = int(interval_time / self.dt) #timepoints
        self.pre_post = pre_post
        self.post_pre = post_pre
        self.reward_pre_post = reward_pre_post
        self.reward_post_pre = reward_post_pre

    def __call__(self, reward):
        processed_neurons = set()
        for neuron in self.group.neurons:
            for preSN in self.network.predecessors(neuron):
                if preSN not in processed_neurons:
                    self._update_synapse(preSN, neuron, reward)

            for postSN in self.network.successors(neuron):
                if postSN not in processed_neurons:
                    self._update_synapse(neuron, postSN, reward)

            processed_neurons.add(neuron)

    def _update_synapse(self, preSN, postSN, reward):
        for timestep in range(self.group.total_timepoints):
            if preSN.spike_train[timestep]:
                if postSN.spike_train[timestep+1:timestep+self.interval+1].any():
                    self._update_weight("pre-post", preSN, postSN, reward)
            if postSN.spike_train[timestep]:
                if preSN.spike_train[timestep+1:timestep+self.interval+1].any():
                    self._update_weight("post-pre", preSN, postSN, reward)

    def _update_weight(self, mode, preSN, postSN, reward):
        if mode == "pre-post":
            if reward:
                self.network.edges.get((preSN, postSN))['weight'] +=\
                np.array([self.reward_pre_post]) 
            else:
                self.network.edges.get((preSN, postSN))['weight'] +=\
                np.array([self.pre_post]) 

        elif mode == "post-pre":
            if reward:
                self.network.edges.get((preSN, postSN))['weight'] +=\
                np.array([self.reward_post_pre]) 
            else:
                self.network.edges.get((preSN, postSN))['weight'] +=\
                np.array([self.post_pre]) 

        else:
            raise ValueError("'mode' must be 'pre-post' or 'post-pre'")