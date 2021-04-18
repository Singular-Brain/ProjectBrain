from explicit.Simulation import Simulation
from explicit.Learning import RFSTDP
from explicit.Elements import IF
#
# 
import numpy as np
from matplotlib import pyplot as plt
###
if __name__ == '__main__':
    sim = Simulation(total_time= 0.1, dt = 0.001)
    group = sim.NeuronGroup(100, connection_chance= 9/10,
                            online_learning_rule = None,
                            neuron_model = IF,
                            save_gif = False,
                            base_current = 1E-9,
                            neuron_attrs = {'save_history': True})
    # Stimulus
    stim1 = sim.Stimulus(lambda t : 1E-9)
    ### Input - output neurons
    input_neurons, output_neurons = group.get_input_output(1, 1)
    # Connect
    stim1.connect(input_neurons[0])
    ### Run
    sim.run()
    ### Learning
    learning = RFSTDP(group)
    learning(reward = True)
    ### Visualization
    group.display_spikes()
    fig, ax = group.draw_graph(display_ids= True)
    plt.show()
