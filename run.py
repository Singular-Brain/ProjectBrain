import os
import numpy as np
from main.neuronTypes.LIF import NeuronGroup
from main.networks import Network, RandomConnections
from main.learningRules.stdp import STDP
from main.callbacks import TensorBoard

dt = 0.001
class Model(Network):
    def architecture(self):
        self.layer1 = RandomConnections(5, 1)
        self.layer2 = RandomConnections(3, 1)
        self.randomConnect(self.layer1, self.layer2, 0.5)


def exp1_reward_function(self,):
    min_, max_ = 1,3
    target_time =  self.timepoint + np.random.randint(min_/dt, max_/dt)
    if target_time < len( self.reward):
        self.reward[target_time] = 0.5

G = NeuronGroup(network= Model(), dt= dt,
                total_time = 1,
                learning_rule = STDP(),
                callbacks = [TensorBoard()] ,
                biological_plausible = True,
                reward_function = exp1_reward_function,
                stochastic_spikes = True,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt*1),
                save_history = False,
                )


G.run(progress_bar = True)