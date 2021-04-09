from RL import *
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 
from Elements import NeuronGroup
from Simulation import Simulation
from Learning import RFSTDP
#
from tqdm import tqdm

class KArmedBandit(Environment):
    def __init__(self):
        self.action_space = {0,1}
        self.reward_space = [0,1]

    def calculate_reward(self, action):
        assert action in self.action_space
        self.reward = self.reward_space[action]

    def step(self, action):
        self.calculate_reward(action)
        observations = None
        return observations, self.reward

class KArmedAgent(Agent):
    def take_action(self):
        action = self.brain.decide()
        _, r = self.env.step(action)
        self.brain.learn(r)
        return r


class KArmedBrain(Brain):
    def __init__(self, dt, engine, lifetime):
        super().__init__(engine, lifetime)
        self.dt = dt
        self.input_neurons, self.output_neurons = engine.get_input_output(1, 2)
        self.learning = RFSTDP(engine)
        # simulation
        self.sim = Simulation(total_time = self.lifetime, dt = self.dt)
        self.sim.add(engine)
        self.stim1 = self.sim.Stimulus(lambda t : 5000)
        self.stim1.connect(self.input_neurons[0])

    def decode_output(self):
        N1Spikes = self.output_neurons[0].spike_train.sum()
        N2Spikes = self.output_neurons[1].spike_train.sum()
        print(f'\n1: {N1Spikes}, 2: {N2Spikes}')
        maximum = max(N1Spikes, N2Spikes)
        if maximum == N1Spikes:
            return 0
        else:
            return 1

    def decide(self):
        self.sim.reset()
        self.sim.run(verbose = True) # Pass stimulus to the engine and run the Simulation
        action = self.decode_output() # Decode engine's output to "action"
        return action

    def learn(self, reward):
        self.learning(reward = reward)


dt = 0.001
total_time = 1 #s
total_timepoints = int(total_time/ dt)

engine = NeuronGroup(500, dt = dt,
                        total_timepoints =total_timepoints,
                        connection_chance= 9/10,
                        online_learning_rule = None,
                        save_gif = False,
                        base_current = 500)

brain = KArmedBrain(dt = dt, engine= engine, lifetime = total_time)
env = KArmedBandit()
agent = KArmedAgent(brain= brain, environment= env )

rewards = []
for _ in (range(10)):
    rewards.append(agent.take_action())

print(rewards)