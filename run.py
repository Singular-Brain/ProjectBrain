import os
import numpy as np
from main.networks import Network
from main.stimulus import frequency_based_current
from main.learningRules.stdp import STDP
from main.callbacks import TensorBoard
from main.neuronTypes import LIF

dt = 0.001
class Model(Network):
    def architecture(self):
        scale_weight = 0.5
        layer1 = self.randomConnections(50, LIF(),  1, excitatory_ratio = 1)
        layer2 = self.randomConnections(30, LIF(), .1, scale_factor = scale_weight)
        layer3 = self.randomConnections(30, LIF(), .1, scale_factor = scale_weight)
        self.randomConnect(layer1, layer2, .2)
        self.randomConnect(layer2, layer3, .05)
        self.connectStimulus(layer1)
        self.connectStimulus(layer2)


# tensorboard = TensorBoard(update_secs = 1)

model = Model(dt= dt,
              total_time = 4,
              learning_rule = STDP(dopamine_base = 0.1),
              callbacks = [],
              save_history = True,
                )

stimuli = [
    {
    frequency_based_current(dt, frequency =  5, amplitude = 1, neurons = [0]),
    frequency_based_current(dt, frequency = 10, amplitude = 1, neurons = [1]),
    frequency_based_current(dt, frequency = 15, amplitude = 1, neurons = [2]),
    frequency_based_current(dt, frequency = 20, amplitude = 1, neurons = [3]),
    frequency_based_current(dt, frequency = 15, amplitude = 1, neurons = [4])
    },
    {
    frequency_based_current(dt, frequency =  5, amplitude = 1, neurons = [0]),
    }
]


model.run(stimuli, progress_bar = True)