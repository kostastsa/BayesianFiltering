class Simulation:

    def __init__(self, model, T, init_state):
        self.model = model  # StateSpaceModel or SLDS
        self.states, self.observs = model.simulate(T, init_state)