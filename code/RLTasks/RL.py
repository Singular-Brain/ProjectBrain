from abc import abstractmethod
from itertools import count

class Environment(object):
    def __init__(self, action_space, state_space):
        self.action_space = action_space
        self.state_space = state_space
        self.state = None
        self.goal_state = None
        self.reset()

    def step(self, action):
        reward = self.calculate_reward(action)
        done = self.is_done()
        self.state = self.next_state(action)
        observations = self.state
        return observations, reward, done
    
    @abstractmethod
    def calculate_reward(self,action):
        pass

    @abstractmethod
    def next_state(self, action):
        pass
        
    #MNote: delete this function:    
    @abstractmethod
    def is_done(self,):
        pass

    #MNote: Chage it to setup
    @abstractmethod
    def reset(self,):
        pass


class Agent(object):
    _ids = count(0)
    def __init__(self, brain, environment):
        self.id = next(self._ids)
        self.brain = brain
        self.env = environment
        self.obs = None
        
    def take_action(self):
        action = self.brain.decide(self.env.state)
        obs, r, done = self.env.step(action)
        self.brain.learn(r)
        if done:
            self.env.reset()
    
    @abstractmethod
    def encode_observation(self, obs):
        pass


class Brain(object):
    def __init__(self, engine, lifetime):
        self.engine = engine
        self.lifetime = lifetime

    @abstractmethod
    def encode_input(self, inp, dt=0.001):
        pass

    @abstractmethod
    def decode_output(self, out):
        pass

    @abstractmethod
    def decide(self, observation):
        self.encode_input(observation) # Encode the observation to stimulus
        self.engine # Pass stimulus to the engine
        self.decode_output(engine_output) # Decode engine's output to "action"
        #return action

    @abstractmethod
    def learn(self, reward):
        pass

    