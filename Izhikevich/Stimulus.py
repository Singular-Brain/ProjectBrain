class Stimulus:
    def __init__(self, output, neurons):
        self.output = output
        self.neurons = neurons

    def __call__(self, timestep):
        return self.output(timestep)