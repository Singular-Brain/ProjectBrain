from explicit.Simulation import Simulation 
from vectorized.main import *
from timeit import timeit
TOTAL_TIME = 1 #s
DT = 0.001
POPULATION_SIZE = 1000
BASE_CURRENT = 10000
CONNECTION_CHANCE = 1
REPEAT_TEST = 10
### Explicit
sim = Simulation(total_time= TOTAL_TIME, dt = DT)
group = sim.NeuronGroup(population_size = POPULATION_SIZE,
                        connection_chance= CONNECTION_CHANCE,
                        online_learning_rule = None,
                        save_gif = False,
                        base_current = BASE_CURRENT)
stim1 = sim.Stimulus(lambda t : 10000)
stim2 = sim.Stimulus(lambda t : 200000 * t)
stim3 = sim.Stimulus(lambda t : 200000 * np.sin(500*t))
input_neurons, output_neurons = group.get_input_output(4, 1)
stim1.connect(input_neurons[0])
stim1.connect(input_neurons[1])
stim2.connect(input_neurons[2])
stim3.connect(input_neurons[3])
print('Calculating explicit mode time...')
explicit_time = timeit('sim.run(verbose = 0)', number = REPEAT_TEST, globals=globals())
### Vectorized
stimuli = {Stimulus(0.001, lambda t: 10000, [0,1]),
           Stimulus(0.001, lambda t: 200000 * t, [2]),
           Stimulus(0.001, lambda t: 200000 * np.sin(500*t), [3])}
G = NeuronGroup(dt = DT, 
                population_size = POPULATION_SIZE, 
                connection_chance = CONNECTION_CHANCE,
                total_time = TOTAL_TIME,
                base_current = 10000,
                stimuli = stimuli)

print('Calculating vectorized mode time...')
vectorized_time = timeit('G.run()', number = REPEAT_TEST, globals=globals())

###
print(f"""Explicit run time:{explicit_time}
Vectorized run time: {vectorized_time}
""")