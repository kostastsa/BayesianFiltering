import jax.numpy as jnp

class ExperimentResults:

    def __init__(self):
        
        self.results = []

    def make_jnp(self):

        self.results = jnp.array(self.results)


class LGSF_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'L-GSF'
        self.marker = 'o'
        self.color = 'b'


class UGSF_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'U-GSF'
        self.marker = 's'
        self.color = 'g'

class LAGSF_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'L-AGSF'
        self.marker = 'D'
        self.color = 'r'

class UAGSF_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'U-AGSF'
        self.marker = 'x'
        self.color = 'm'

class BPF_Results(ExperimentResults):

    def __init__(self):
        super().__init__()
        self.name = 'BPF'
        self.marker = 'v'
        self.color = 'c'

class APF_Results(ExperimentResults):
    
    def __init__(self):
        super().__init__()
        self.name = 'APF'
        self.marker = '^'
        self.color = 'y'