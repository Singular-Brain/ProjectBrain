import os
import numpy as np
from main import NeuronGroup
from AdjacencyMatrix import uniform_connections

dt = 0.001
DEVICE = 'cpu'

network = uniform_connections(1000, connection_chance = 0.1, excitatory_ratio = 0.8)
network[0,1] = 0.01
initial_network = network.copy()

def exp1_reward_function(dt, spike_train, spikes, timepoint, reward):
    min_, max_ = 1,3
    presynaptic_neuron, postsynaptic_neuron = 0, 1
    if spikes[postsynaptic_neuron] and\
        spike_train[presynaptic_neuron, timepoint-10:timepoint].any():
        target_time = timepoint + np.random.randint(min_/dt, max_/dt)
        if target_time < len(reward):
            reward[target_time] = 0.5
    return reward



def setup_online_plot(self):
    pass
    # target_weights = []
    # plot_times = []
    # fig, axs = plt.subplots(3,figsize=(20,20))
    # plt.ion()

def update_online_plot(self):
    pass
    # if self.timepoint == self.total_timepoints-1 or self.timepoint%1000==0:
    #     target_weights.append(self.weights[0][1].cpu())
    #     plot_times.append(self.timepoint/1000)
    #     # y, x = np.where(self.spike_train[:,:self.timepoint].cpu())
    #     #axs[0].clear()
    #     axs[1].clear()
    #     axs[2].clear()
    #     #spike train
    #     # axs[0].plot(x/1000, y, '.', color='black', marker='o', markersize=100/np.sqrt(len(x)))
    #     # axs[0].set_xlim([0,self.timepoint/1000])
    #     # axs[0].set_ylim([0,self.N])
    #     # weights histogram
    #     axs[1].hist(self.weights.cpu()[self.weights>0], bins = 100)
    #     #target synapse's weight and rewarsd
    #     axs[2].plot(plot_times, target_weights,)
    #     rewards = np.where(self.reward.cpu() > 0.1)[0]
    #     axs[2].plot(rewards/1000, np.zeros_like(rewards), 'r*')
        #axs[2].set_ylim([-0.1,4.1])
        ### local:
        #fig.canvas.draw()
        #fig.canvas.flush_events()
        ### google colab
        # display.clear_output(wait=True)
        # display.display(plt.gcf())

        # for i in range(6):
        #     axs[i].clear()
        # for i, (x, label) in enumerate(zip([pr,pstdp,pd,pet,pw], ['Reward', 'STDP','Dopamine', 'Eligibity trace', 'weight'])):
        #     axs[i].plot(x)
        #     axs[i].set_xlabel(label)
        # axs[5].clear()
        # axs[5].hist(self.weights.cpu()[self.weights > 0], bins = 100)
        # display.clear_output(wait=True)
        # display.display(plt.gcf())


G = NeuronGroup(network= network, dt= dt,
                total_time = 3600,
                stimuli = set(),
                biological_plausible = True,
                neuron_type = "IZH", 
                stochastic_spikes = True,
                reward_function = exp1_reward_function,
                plastic_inhibitory = False,
                stochastic_function_b = 1/0.013,
                stochastic_function_tau = (np.exp(-1))/(dt*1),
                save_history = False,
                process_bar = False,
                )


G.run()
# G.plot_spikes()