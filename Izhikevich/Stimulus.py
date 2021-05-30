class Stimulus:
    def __init__(self, output, neurons):
        self.output = output
        self.neurons = neurons

    def __call__(self, timestep):
        return self.output(timestep)

class frequency_based_current:
    def __init__(self, dt, frequency, amplitude, neurons):
        assert dt * frequency <= 1, "Frequency is too high to be implemented with the value of 'dt'"
        self.dt = dt
        self.step = round(1/(frequency * dt))
        self.amplitude = amplitude
        self.neurons = neurons

    def __call__(self, timestep):
        if timestep%self.step==0:
            return self.amplitude
        return 0