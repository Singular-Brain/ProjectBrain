from main import *
from Visualization import NetworkPanel

# Reproducibility
#manual_seed(2045)

POPULATION_SIZE = 100
DECODING_METHOD = 'TIME_TO_FIRST_SPIKE'
#DECODING_METHOD = 'RATE_CODING'

stimuli = {
        Stimulus(0.001, lambda t: .6E-5, list(range(POPULATION_SIZE//10)))\
        #,Stimulus(0.001, lambda t: 20E2 * t, [2]),
        #Stimulus(0.001, lambda t: 1E2 * np.sin(500*t), [3])
        }

G = NeuronGroup(dt = 0.001, population_size = POPULATION_SIZE, connection_chance = 1/3,
                total_time = 1, stimuli = stimuli, neuron_type='IZH', biological_plausible=True, scaled_variables = False)

learning = RFSTDP(G, interval_time = 0.004, # seconds
                 pre_post_rate = 0.001,
                 reward_pre_post_rate = 0.002,
                 post_pre_rate = -0.001,
                 reward_post_pre_rate = 0.001,
                 )
G.run()
#G.display_spikes()
#print(G.spike_train.sum(axis = 1))

for _ in range(50):
    G.run()
    if DECODING_METHOD == 'RATE_CODING':
        output = G.spike_train[-2:].sum(axis = 1)
    elif DECODING_METHOD == 'TIME_TO_FIRST_SPIKE':
        output = [0,0]
        #print(G.spike_train[-2])
        #print(G.spike_train[-1])
        output[0] = -np.where(G.spike_train[-2]==True)[0][0]
        output[1] = -np.where(G.spike_train[-1]==True)[0][0]
    print(output)
    if output[0] > output[1]:
        reward = True
    else:
        reward = False
    learning(reward = reward)
