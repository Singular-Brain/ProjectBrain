import os
import numpy as np
from Visualization import NetworkPanel
from main import *
from AdjacencyMatrix import *

stimuli = {
        Stimulus(0.001, lambda t: 1, [0]),
        # Stimulus(0.001, lambda t: 20 * t, [2]),
        # Stimulus(0.001, lambda t: 1 * np.sin(500*t), [3])
        }


network = random_connections(5, connection_chance = 1, excitatory_chance = 1)

def exp1_reward_function(dt, spike_train, timepoint, reward):
    presynaptic_neuron, postsynaptic_neuron = 0, 1
    if spike_train[postsynaptic_neuron, timepoint] and\
        spike_train[presynaptic_neuron, timepoint-10:timepoint].any():
        reward[np.random.randint(1/dt, 3/dt)] = 0.5
    return reward

G = NeuronGroup(network=network, dt = 0.001, total_time = 0.1, stimuli = stimuli,
                biological_plausible = True,
                reward_function = exp1_reward_function,
                save_history = True,
                save_to_file = False)


G.run()
G.display_spikes()