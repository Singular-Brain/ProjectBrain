from Visualization import NetworkPanel
from main import *
from AdjacencyMatrix import random_connections, recurrent_layer_wise

stimuli = {
        Stimulus(0.001, lambda t: 1, [0,1]),
        Stimulus(0.001, lambda t: 20 * t, [2]),
        Stimulus(0.001, lambda t: 1 * np.sin(500*t), [3])
        }

# network = random_connections(100, connection_chance = 0.1, excitatory_chance = 0.8)
network = recurrent_layer_wise([4, 20, 20, 20, 10, 2], recurrent_connection_chance = .05, between_connection_chance = 0.8, inside_connection_chance = 0.2, excitatory_chance = 0.8)
G = NeuronGroup(network = network, dt = 0.001, total_time = 0.1, stimuli = stimuli,
                base_current= 1,
                u_thresh= 1,
                u_rest= -0,
                tau_refractory= 0.005,
                excitatory_chance=  0.8,
                Rm= 5,
                Cm= 0.001,
                save_history = True,)

G.run()
G.display_spikes()
import os
panel = NetworkPanel(G, input_neurons = range(4), file_path = __file__)
panel.display()
print(G.weights)
learning = RFSTDP(G)
learning(reward = True)
print(G.weights)
