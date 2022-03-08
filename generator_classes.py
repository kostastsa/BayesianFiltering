import numpy as np

class LinearModelParameters:

    def __init__(self, A ,H, Q, R):
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

    def __str__(self):

        return 'A : \n' + str(self.A) + '\n' + 'H : \n' + str(self.H) + '\n' + 'Q : \n' + str(self.Q) + '\n' + 'R : \n' + str(self.R)

class StateSpaceModel:
    """
    Generative Model class.
    
    Attributes
    ----------
    """

    def __init__(self, dx, dy, f = None, g = None, descr = None):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.f = f
        self.g = g
        self.descr = descr

    def __str__(self):
        if self.descr == "LG":
            return str(self.params)

    def simulate(self, T, init_state):
        self.T = T
        states = np.zeros([T, self.dx])
        observs = np.zeros([T, self.dy])
        prev_state = init_state
        for t in range(T):
            new_state = self.f(prev_state)
            new_obs = self.g(new_state)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return states, observs

    def propagate(self, prev_state):
        new_state = self.f(prev_state)
        new_obs = self.g(new_state)
        return new_state, new_obs



class LGSSM(StateSpaceModel):

    def __init__(self, dx, dy, parameters: LinearModelParameters):
        # TODO: fix case where dy == 1 but dx > 1
        # TODO: separate functions f and g from noises
        self.dx = dx
        self.dy = dy
        self.descr = "LG"
        self.params = parameters
        if self.dx == 1:
            self.f = lambda prev_state: self.params.A * prev_state + \
                                        np.random.normal(np.zeros(1), self.params.Q)
        else:
            self.f = lambda prev_state: np.matmul(prev_state, self.params.A.T) + \
                                        np.random.multivariate_normal(np.zeros([self.dx]), self.params.Q)
        if self.dy == 1:
            self.g = lambda prev_state: self.params.H * prev_state + \
                                        np.random.normal(np.zeros(1), self.params.R)
        else:
            self.g = lambda state: np.matmul(state, self.params.H.T) + \
                                   np.random.multivariate_normal(np.zeros([self.dy]), self.params.R)



class SLDS:
    """
    Regime Switching State Space Model class.

    Attributes
    ----------
    """
    
    def __init__(self, dx, dy, model_parameter_array):
        """Initialization method."""
        self.dx = dx
        self.dy = dy
        self.num_models = len(model_parameter_array)
        self.transition_matrix = np.zeros([self.num_models, self.num_models])
        self.models = np.empty([self.num_models], dtype = StateSpaceModel)
        for m in range(self.num_models):
            self.models[m] = LGSSM(dx,dy,model_parameter_array[m])

    def set_transition_matrix(self, mat):
        self.transition_matrix = mat

    def simulate(self, T, init_state):
        model_history = np.zeros([T], dtype = int)
        states = np.zeros([T, self.dx], dtype=float)
        observs = np.zeros([T, self.dy], dtype=float)

        prev_model = init_state[0]
        prev_state = init_state[1]

        for t in range(T):
            new_model = np.random.choice(range(self.num_models), p=self.transition_matrix[prev_model, :])
            model_history[t] = new_model
            prev_model = new_model
            new_state, new_obs = self.models[new_model].propagate(prev_state)
            states[t, :] = new_state
            observs[t, :] = new_obs
            prev_state = new_state
        return model_history, states, observs




class Simulation:

    def __init__(self, model, T, init_state):
        self.model = model # StateSpaceModel or SLDS
        self.all_data = model.simulate(T, init_state)