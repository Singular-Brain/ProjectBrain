import os
import numpy as np
from NeuronTypes.LIF import NeuronGroup
from AdjacencyMatrix import uniform_connections
from Stimulus import Stimulus
from matplotlib import pyplot as plt

dt = 0.001

network = uniform_connections(1000, connection_chance = 0.1, excitatory_ratio = 0.8)
network[0,1] = 0.01
initial_network = network.copy()

stimulus = {
        Stimulus(dt, lambda t: 1 if t%10==0 else 0, range(50)),
        }

def exp1_reward_function(dt, spike_train, spikes, timepoint, reward):
    if timepoint%10000 == 20: # 20ms after stimulus injection
        groupA = spike_train[50:100, timepoint-20: timepoint].sum()
        groupB = spike_train[100:150, timepoint-20: timepoint].sum()
        if groupA > groupB:
            reward[timepoint  + np.random.randint(0, 1/dt)] = 0.5
    return reward


def setup_online_plot(self):
    self.groupA = []
    self.groupB = []
    fig, axs = plt.subplots(2,figsize=(20,10))
    plt.ion()
    plt.show()
    return fig, axs

def update_online_plot(self, fig, axs):
    if self.timepoint%10000 == 20: # 20ms after stimulus injection
        self.groupA.append(self.spike_train[50:100, self.timepoint-20: self.timepoint].sum())
        self.groupB.append(self.spike_train[100:150, self.timepoint-20: self.timepoint].sum())
        for i in range(2):
            axs[i].clear()
        x,y = np.where(self.spike_train[:150, self.timepoint-20: self.timepoint].cpu())
        axs[0].plot(y,x, '.',  color='black', markersize=5, alpha = 0.8 )
        axs[0].set_xlim((0,20))
        axs[0].set_ylim((0,150))
        axs[0].plot((0,20),(50,50), color='blue', markersize=0.5, alpha = 0.5)
        axs[0].plot((0,20),(100,100), color='blue', markersize=0.5, alpha = 0.5)
        axs[0].set_xlabel(f'Trial {(self.timepoint//10000) + 1}')
        ###
        axs[1].plot(self.groupA)
        axs[1].plot(self.groupB)
        ### local:
        fig.canvas.draw()
        fig.canvas.flush_events()
        ### Notebook:
        # display.clear_output(wait=True)
        # display.display(plt.gcf())


G = NeuronGroup(network= network, dt= dt,
                total_time = 1000,
                stimuli = stimulus,
                biological_plausible = True,
                stochastic_spikes = True,
                reward_function = exp1_reward_function,
                plastic_inhibitory = False,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt*1),
                save_history = False,
                process_bar = True,
                setup_online_plot = setup_online_plot,
                update_online_plot = update_online_plot,
                )


G.run()