import os
import numpy as np
from main.neuronTypes.LIF import NeuronGroup
from main.adjacencyMatrix import random_connections
from main.learningRules.stdp import STDP
from main.callbacks import TensorBoard

dt = 0.001
network = random_connections(1000, connection_chance = 0.1, excitatory_ratio = 0.8)
initial_network = network.copy()

def exp1_reward_function(self,):
    min_, max_ = 1,3
    target_time =  self.timepoint + np.random.randint(min_/dt, max_/dt)
    if target_time < len( self.reward):
        self.reward[target_time] = 0.5

stdp = STDP()
G = NeuronGroup(network= network, dt= dt,
                total_time = 1,
                learning_rule = stdp,
                callbacks = [TensorBoard()] ,
                biological_plausible = True,
                reward_function = exp1_reward_function,
                stochastic_spikes = True,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt*1),
                save_history = False,
                )


G.run(progress_bar = True)