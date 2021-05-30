
import os
import numpy as np
from dataloader import train_loader, test_loader
from NeuronTypes.LIF import NeuronGroup
from AdjacencyMatrix import recurrent_layer_wise
from Stimulus import Stimulus
from matplotlib import pyplot as plt

dt = 0.001

network = recurrent_layer_wise([28*28, 512, 100], recurrent_connection_chance = 0.02,
    between_connection_chance = [0.1, 0.1], inside_connection_chance = [0, 0.1, 0.05],
    excitatory_ratio = 0.8)
initial_network = network.copy()

def exp1_reward_function(self, spikes):
    min_, max_ = 1,3
    presynaptic_neuron, postsynaptic_neuron = 0, 1
    if spikes[postsynaptic_neuron] and\
         self.spike_train[presynaptic_neuron,  self.timepoint-10: self.timepoint].any():
        target_time =  self.timepoint + np.random.randint(min_/dt, max_/dt)
        if target_time < len( self.reward):
             self.reward[target_time] = 0.5
    return  self.reward


def setup_online_plot(self):
    self.target_weights = []
    fig, axs = plt.subplots(2,figsize=(20,20))
    plt.ion()
    plt.show()
    return fig, axs

def update_online_plot(self, fig, axs):
    self.target_weights.append(self.weights[0][1].cpu())
    if self.timepoint == self.total_timepoints-1 or self.timepoint%1000==0:
        ### clear axs
        for i in range(2):
            axs[i].clear()
        ### weights histogram
        axs[0].hist(self.weights.cpu()[self.weights>0], bins = 100)
        axs[0].set_xlabel("weights")
        ###target synapse's weight and rewarsd
        axs[1].plot(self.target_weights)
        rewards = np.where(self.reward.cpu() > 0.1)[0]
        axs[1].plot(rewards, np.zeros_like(rewards), 'r*')
        axs[1].set_xlabel("Target synapse weight (*: rewards)")
        ### local:
        fig.canvas.draw()
        fig.canvas.flush_events()
        ### Notebook:
        # display.clear_output(wait=True)
        # display.display(plt.gcf())


sample_image = next(iter(train_loader))[0][0]
stimuli = set()
for neuron, pixel in enumerate(sample_image.ravel()):
    stimuli.add(Stimulus(lambda t: pixel, neuron))

G = NeuronGroup(network= network, dt= dt,
                total_time = 1,
                stimuli = stimuli,
                online_learning = True,
                biological_plausible = True,
                stochastic_spikes = True,
                reward_function = None,
                plastic_inhibitory = True,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt*1),
                save_history = False,
                process_bar = True,
                setup_online_plot = setup_online_plot,
                update_online_plot = update_online_plot,
                )


G.run()