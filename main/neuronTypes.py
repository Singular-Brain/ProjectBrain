
class LIF:
    def __init__(self, Cm = 14.7E-12, Rm= 135E6, u_thresh = -55E-3,
                 u_rest = -68E-3, tau_refractory = 0.002):
        self.Cm = Cm
        self.tau_m = self.Cm * Rm 
        self.u_thresh = u_thresh
        self.u_rest = u_rest
        self.tau_refractory = tau_refractory 

    def __call__(self,dt, potential, current):
        return (-dt/self.tau_m)*(potential-self.u_rest) + dt * current / self.Cm
