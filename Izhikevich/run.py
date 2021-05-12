import os
import numpy as np
from Visualization import NetworkPanel
from main import *
from AdjacencyMatrix import *

dt = 0.001

stimuli = {
        Stimulus(dt, lambda t: 1, [0]),
        # Stimulus(dt, lambda t: 20 * t, [2]),
        # Stimulus(dt, lambda t: 1 * np.sin(500*t), [3])
        }


network = random_connections(100, connection_chance = 0.1, excitatory_ratio = 0.8)

def exp1_reward_function(dt, spike_train, timepoint, reward):
    presynaptic_neuron, postsynaptic_neuron = 0, 1
    if spike_train[postsynaptic_neuron, timepoint] and\
        spike_train[presynaptic_neuron, timepoint-10:timepoint].any():
        reward[np.random.randint(1/dt, 3/dt)] = 0.5
    return reward

G = NeuronGroup(network= network, dt= dt,
                total_time = 0.5,
                stimuli = set(),
                biological_plausible = True,
                neuron_type = "LIF", 
                stochastic_spikes = True,
                reward_function = exp1_reward_function,
                plastic_inhibitory = False,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/dt,
                save_history = True,
                save_to_file = False)


G.run()
G.plot_spikes()