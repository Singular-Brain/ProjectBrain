from main import *
from Visualization import NetworkPanel

# Reproducibility
#manual_seed(2045)

#POPULATION_SIZE = 1000
#DECODING_METHOD = 'TIME_TO_FIRST_SPIKE'
DECODING_METHOD = 'RATE_CODING'

stimuli = {
        Stimulus(0.001, lambda t: 1, [0,1]),
        Stimulus(0.001, lambda t: 20 * t, [2]),
        Stimulus(0.001, lambda t: 1 * np.sin(500*t), [3])
        }

# based on Neural dynamics P5, each neuron connects to more than 10^4 postsynaptic neurons,
# and we have ~ 10^11 neurons. spikes has 100 mv apmplitude, and a duration of 1-2 ms.
network = recurrent_layer_wise([4, 20, 50, 100, 20, 2], recurrent_connection_chance = .05, between_connection_chance = 0.4, inside_connection_chance = 0.2, excitatory_chance = 0.8)

G = NeuronGroup(network=network, dt = 0.001, population_size = 100, connection_chance = 0.1, total_time = 0.1, stimuli = stimuli,
                base_current= 1,
                u_thresh= 1,
                u_rest= -0,
                tau_refractory= 0.005,
                excitatory_chance=  0.8,
                Rm= 5,
                Cm= 0.001,
                save_history = True,)

learning = RFSTDP(G, interval_time = 0.004, # seconds
                 pre_post_rate = 0.001,
                 reward_pre_post_rate = 0.002,
                 post_pre_rate = -0.001,
                 reward_post_pre_rate = 0.001,
                 )
G.run()
G.display_spikes()
#print(G.spike_train.sum(axis = 1))

for _ in range(50):
    G.run()
    #print(G.spike_train.sum(axis = 1))
    if DECODING_METHOD == 'RATE_CODING':
        output = G.spike_train[-2:].sum(axis = 1)
    elif DECODING_METHOD == 'TIME_TO_FIRST_SPIKE':
        output = [0,0]
        #print(G.spike_train[-2])
        #print(G.spike_train[-1])
        if len(torch.where(G.spike_train[-2]==True)[0]) == 0:
            output[0] = -np.inf
        else:
            output[0] = -torch.where(G.spike_train[-2]==True)[0][0].detach().cpu().numpy()
        if len(torch.where(G.spike_train[-1]==True)[0]) == 0:
            output[1] = -np.inf
        else:
            output[1] = -(torch.where(G.spike_train[-1]==True)[0][0]).detach().cpu().numpy()
    print(output)
    if output[0] > output[1]:
        reward = True
    else:
        reward = False
    learning(reward = reward)
