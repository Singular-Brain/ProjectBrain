import os
import numpy as np
from main import *
from AdjacencyMatrix import *

dt = 0.001

network = uniform_connections(100, connection_chance = 0.2, excitatory_ratio = 0.8)
network[0,1] = 0.01
initial_network = network.copy()

def exp1_reward_function(dt, spike_train, spikes, timepoint, reward):
    presynaptic_neuron, postsynaptic_neuron = 0, 1
    if spikes[postsynaptic_neuron] and\
        spike_train[presynaptic_neuron, timepoint-10:timepoint].any():
        target_time = timepoint + np.random.randint(1/dt, 3/dt)
        if target_time < len(reward):
            reward[target_time] = 0.5
    return reward

G = NeuronGroup(network= network, dt= dt,
                total_time = 15,
                stimuli = set(),
                biological_plausible = True,
                neuron_type = "LIF", 
                stochastic_spikes = True,
                reward_function = exp1_reward_function,
                plastic_inhibitory = False,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt * 15),
                save_history = False,
                process_bar = True,
                )


G.run()
G.plot_spikes()