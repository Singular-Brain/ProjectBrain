from Simulation import Simulation
from Learning import RFSTDP
#
# 
import numpy as np
from matplotlib import pyplot as plt
###
if __name__ == '__main__':
    sim = Simulation(total_time= 0.1, dt = 0.001)
    group = sim.NeuronGroup(500, connection_chance= 9/10,
                            online_learning_rule = None,
                            save_gif = False,
                            base_current = 500)
    # Stimulus
    stim1 = sim.Stimulus(lambda t : 5000)
    stim2 = sim.Stimulus(lambda t : 5000 *np.sin(5*t))
    stim3 = sim.Stimulus(lambda t : 5000 * np.log(t + 1))
    # Input - output neurons
    input_neurons, output_neurons = group.get_input_output(3, 1)
    # Connect
    stim1.connect(input_neurons[0])
    stim2.connect(input_neurons[1])
    stim3.connect(input_neurons[2])
    # Run
    sim.run()
    # Learning
    learning = RFSTDP(group)
    learning(reward = True)
    # Visualization
    group.display_spikes()
    # fig, ax = group.draw_graph(display_ids= True)
    # plt.show()
