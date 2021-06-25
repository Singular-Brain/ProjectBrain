import numpy as np
from main.networks import Network
from main.stimulus import frequency_based_current
from main.learningRules.stdp import STDP
from main.callbacks import Callback, TensorBoard
from main.neuronTypes import LIF

dt = 0.001
stochastic_params = {'stochastic_spikes': True,
                     'stochastic_function_b': 1/0.013,
                     'stochastic_function_tau': (np.exp(-1))/(dt*1)}
class Model(Network):
    def architecture(self):
        scale_weight = 0.5
        main = self.randomConnections(100, LIF(),  1, scale_factor = scale_weight, **stochastic_params,)
        label0 = self.randomConnections(50, LIF(), 0, name = '0', **stochastic_params)
        label1 = self.randomConnections(50, LIF(), 0, name = '1', **stochastic_params)
        self.randomConnect(main, label0, .1)
        self.randomConnect(main, label1, .1)
        # self.connectStimulus(main)


class Reward(Callback):
    def __init__(self, n_classes, release_dopamine_per_spike):
        super().__init__()
        self.n_classes = n_classes
        self.release_dopamine = release_dopamine_per_spike

    def set_label(self, label):
        self.label = label

    def on_subnetwork_learning_start(self, subNetwork, learning_rule, timepoint):
        if subNetwork.name ==str(self.label):
            learning_rule.release_dopamine(self.release_dopamine * subNetwork.spikes.sum())


tensorboard = TensorBoard(update_secs = 1)
reward = Reward(n_classes = 2, release_dopamine_per_spike = 0.004)

model = Model(dt= dt,
              total_time = 5,
              learning_rule = STDP(dopamine_base = 0.001, excitatory_hardbound = None),
              callbacks = [tensorboard, reward],
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
]

reward.set_label(0)
model.run(progress_bar = False)