class Stimulus:
    def __init__(self, output, neurons):
        self.timepoint = 0
        self.output = output
        self.neurons = neurons
        self.generator = self.generator_function()

    def generator_function(self):
        while True:
            yield self.output(self.timepoint)
            self.timepoint +=1

    def __call__(self,):
        return next(self.generator)