from main import *

stimuli = {
        Stimulus(0.001, lambda t: .7E-9, [0,1]),
        Stimulus(0.001, lambda t: 20E-9 * t, [2]),
        Stimulus(0.001, lambda t: 1E-9 * np.sin(500*t), [3])
        }

G = NeuronGroup(dt = 0.001, population_size = 10, connection_chance = 1/2,
                total_time = 1, stimuli = stimuli, base_current = 1E-9)

learning = RFSTDP(G, interval_time = 0.004, # seconds
                 pre_post_rate = 0.001,
                 reward_pre_post_rate = 0.002,
                 post_pre_rate = -0.001,
                 reward_post_pre_rate = 0.001,
                 )
G.run()
G.display_spikes()
print(G.spike_train.sum(axis = 1))

for _ in range(50):
    G.run()
    output = G.spike_train[-2:].sum(axis = 1)
    print(output)
    if output[0] > output[1]:
        reward = True
    else:
        reward = False
    learning(reward = reward)
