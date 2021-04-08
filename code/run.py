from Simulation import Simulation
from Elements import Stimulus
from Learning import RFSTDP
#
# 
import numpy as np
###
sim = Simulation(total_time= 0.1, dt = 0.001)
group = sim.NeuronGroup(30, connection_chance= 2/3,
                        online_learning_rule = None, save_gif = False,
                        base_current = 20000)
# Stimulus
stim1 = sim.Stimulus(lambda t : 5000)
stim2 = sim.Stimulus(lambda t : 5000 *np.sin(5*t))
stim3 = sim.Stimulus(lambda t : 5000 * np.log(t + 1))
# Input neurons
neurons_list = list(group.neurons)
input_neuron1 =  neurons_list[0]
input_neuron2 =  neurons_list[1]
input_neuron3 =  neurons_list[2]
# Connect
stim1.connect(input_neuron1)
stim2.connect(input_neuron2)
stim3.connect(input_neuron3)
# Run
sim.run()
# Learning
learning = RFSTDP(group)
learning(True)