import os
from Visualization import NetworkPanel
from main import *
from AdjacencyMatrix import *

stimuli = {
        Stimulus(0.001, lambda t: 1, [0]),
        # Stimulus(0.001, lambda t: 20 * t, [2]),
        # Stimulus(0.001, lambda t: 1 * np.sin(500*t), [3])
        }


network = recurrent_layer_wise([1, 3, 2], recurrent_connection_chance = .05, between_connection_chance = 0.8, inside_connection_chance = 0.2, excitatory_chance = 0.8, between_connection_chance_decay=0.85)

G = NeuronGroup(network=network, dt = 0.001, population_size = 100, connection_chance = 0.1, total_time = 0.1, stimuli = stimuli,
                base_current= 1,
                u_thresh= 1,
                u_rest= -0,
                tau_refractory= 0.005,
                excitatory_chance=  0.8,
                Rm= 5,
                Cm= 0.001,
                save_history = True,
                save_to_file = 'RunData.npy')

G.run()
G.display_spikes()
panel = NetworkPanel(G, input_neurons = [0], file_path = __file__)
panel.display()
print(G.weights)
learning = RFSTDP(G)
learning(reward = True)
print(G.weights)
