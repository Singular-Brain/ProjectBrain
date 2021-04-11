from .Elements import NeuronGroup, Stimulus
#
import time
# 
from tqdm import tqdm

class Simulation(object):
    def __init__(self, total_time, dt):
        """
        """
        self.total_timepoints = int(total_time/ dt)
        self.timestep = 0
        self.objects = set()
        self.dt = dt

    def NeuronGroup(self, population, **kwargs):
        G = NeuronGroup(population, self.total_timepoints, self.dt, **kwargs)
        self.objects.add(G)
        # if G.online_learning_rule:
        #     self.reward_based = G.online_learning_rule.reward_based
        return G

    def Stimulus(self, output):
        stim = Stimulus(output, self.dt)
        self.objects.add(stim)
        return stim

    def add(self, obj):
        self.objects.add(obj)
        # if isinstance(obj, NeuronGroup):
        #     if G.online_learning_rule:
        #         self.reward_based = G.online_learning_rule.reward_based
                
    def step(self):
        #TODO: define self.sorted_objects to avoid sorting in each step 
        for obj in sorted(self.objects, key= lambda obj: obj.order):
            obj.timestep = self.timestep
            obj.step()
            # if isinstance(obj, NeuronGroup) and obj.online_learning_rule and obj.online_learning_rule.reward_based:
            #     obj.online_learning_rule.set_reward()

    def run(self, verbose = 2):
        self._reset()
        if verbose == 0:
            for self.timestep in range(self.total_timepoints):
                self.step()
        else:
            start_time = time.time()
            for _ in range(self.total_timepoints) if verbose==1 else tqdm(range(self.total_timepoints)):
                self.step()
            run_time = time.time() - start_time
            print(f"Simulation finished in {round(run_time,2)}s")

    def _reset(self):
        for obj in self.objects:
            obj.timestep = 0

    @property
    def second(self):
        return self.timestep * self.dt