
import numpy as np
import torch
from main.networks import Network
from main.stimulus import frequency_based_current
from main.learningRules.stdp import STDP
from main.callbacks import Callback, TensorBoard
from main.neuronTypes import LIF
from MNIST.dataloader import train_loader

N_EPOCHS = 10
dt = 0.001 #s
TOTAL_TIME = 0.5 #s

class Model(Network):
    def architecture(self):
        stochastic_params = {'stochastic_spikes': True,
                     'stochastic_function_b': 1/0.013,
                     'stochastic_function_tau': (np.exp(-1))/(dt*1)}
        scale_weight = 0.5
        inp = self.randomConnections(28 * 28, LIF(), 0, name = 'input', excitatory_ratio = 1)
        main = self.randomConnections(1000, LIF(),  0.1, scale_factor = scale_weight, **stochastic_params,)
        label0 = self.randomConnections(100, LIF(), 0.1, name = '3', **stochastic_params)
        label1 = self.randomConnections(100, LIF(), 0.1, name = '5', **stochastic_params)
        self.randomConnect(inp,  main,   .05)
        self.randomConnect(main, label0, .1)
        self.randomConnect(main, label1, .1)
        self.connectStimulus(inp)


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


tensorboard = TensorBoard()
reward = Reward(n_classes = 2, release_dopamine_per_spike = 0.004)

model = Model(dt= dt,
              total_time = TOTAL_TIME,
              batch_size = train_loader.batch_size,
              learning_rule = STDP(dopamine_base = 0.001,
                    excitatory_hardbound = (0,1), inhibitory_hardbound = (-1,0)),
              callbacks = [tensorboard, reward],
              save_history = True,
                )


for epoch in range(N_EPOCHS):
    print(f'epoch: {epoch}')
    for i, (images, labels) in enumerate(train_loader):
        stimulus = torch.zeros((train_loader.batch_size, 28*28, model.total_timepoints), dtype= torch.float32)
        stimulus[images.squeeze(1).view(train_loader.batch_size, 28*28).bool()] = torch.tensor(([1E-3] + [0] * 49) * 10, dtype = torch.float32)
        reward.set_label(0)
        model.run(stimuli = stimulus, progress_bar = True)