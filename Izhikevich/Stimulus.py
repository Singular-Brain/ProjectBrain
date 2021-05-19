class Stimulus:
    def __init__(self, dt, output, neurons):
        self.output = output
        self.neurons = neurons
        self.dt = dt
 
    def __call__(self, timestep):
        return self.output(timestep * self.dt)
 