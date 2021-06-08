
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
    excitatory_ratio = [0, .8, 0])
initial_network = network.copy()

def reward_function(self, spikes):
    pass

def setup_online_plot(self):
    fig, axs = plt.subplots(3,figsize=(5,5))
    ###local:
    plt.ion()
    plt.show()
    return fig, axs

def update_online_plot(self, fig, axs):
    if self.timepoint == self.total_timepoints-1 or self.timepoint%1000==0:
        ### clear axs
        for i in range(3):
            axs[i].clear()
        ### spike trains:
        x,y = np.where(self.spike_train[:, self.timepoint-1000: self.timepoint].cpu())
        axs[0].plot(y,x, '.',  color='black', markersize=2, alpha = 0.8 )
        axs[0].set_xlim((0,1000))
        axs[0].set_ylim((0,1396))
        axs[0].plot((0,1000),(784,784), color='blue', markersize=0.5, alpha = 0.5)
        axs[0].plot((0,1000),(1296,1296), color='blue', markersize=0.5, alpha = 0.5)
        axs[0].set_xlabel(f'Trial {(self.timepoint//10000) + 1}')
        ### weights histogram
        axs[1].hist(self.weights.cpu()[self.weights>0], bins = 100)
        axs[1].set_xlabel("weights")
        ###rewards
        rewards = np.where(self.reward.cpu() > 0.1)[0]
        axs[2].plot(rewards, np.zeros_like(rewards), 'r*')
        axs[2].set_xlabel("Target synapse weight (*: rewards)")
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
                total_time = 10,
                stimuli = stimuli,
                online_learning = True,
                biological_plausible = True,
                stochastic_spikes = False, #True,
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