# This module is adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios under the MIT License

import os
import re
import util.misc

main_dir = '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/gaussfiltax'

class ParseError(Exception):
    """
    Exception to be thrown when there is a parsing error.
    """

    def __init__(self, str):

        self.str = str

    def __str__(self):

        return self.str


class SimulatorDescriptor:

    @staticmethod
    def get_descriptor(str):

        if re.match('lgssm', str):

            return LGSSM_Descriptor(str)

        elif re.match('lmsv', str):

            return LMSV_Descriptor(str)

        elif re.match('mtt', str):

            return MTT_Descriptor(str)
        
        elif re.match('covinfl', str):

            return COVINFL_Descriptor(str)

        else:

            raise ParseError(str)


class LGSSM_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.name = 'lgssm'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lgssm\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0}\n'.format(self.num_timesteps)
        str += '\t}'

        return str

    def create_desc(self, 
                    state_dim, 
                    emission_dim, 
                    num_timesteps 
                    ):
        
        str = 'sim: lgssm\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0}\n'.format(num_timesteps)
        str += '\n'
        str += '\t}'
    
        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'lgssm\{state_dim:(.*),emission_dim:(.*),num_timesteps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))

    def get_dir(self):

        return os.path.join('lgssm', 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}'.format(self.state_dim, self.emission_dim, self.num_timesteps))


class LMSV_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.name = 'lmsv'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lmsv\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0}\n'.format(self.num_timesteps)
        str += '\t}'

        return str

    def create_desc(self, 
                    state_dim, 
                    emission_dim, 
                    num_timesteps 
                    ):
        
        str = 'sim: lmsv\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0}\n'.format(num_timesteps)
        str += '\n'
        str += '\t}'
    
        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'lmsv\{state_dim:(.*),emission_dim:(.*),num_timesteps:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))

    def get_dir(self):

        return os.path.join('lmsv', 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}'.format(self.state_dim, self.emission_dim, self.num_timesteps))


class COVINFL_Descriptor(SimulatorDescriptor):
    
    def __init__(self, str=None):

        self.state_dim = 1
        self.input_dim = 0
        self.emission_dim = None
        self.num_timesteps = None
        self.a = None
        self.b = None        
        self.target_vars = None
        self.name = 'covinfl'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'covinfl\n'
        str += '\t{\n'
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
        str += '\t\ta: {0},\n'.format(self.a)
        str += '\t\tb: {0}\n'.format(self.b)
        str += '\t}'

        return str
    
    def create_desc(self, state_dim, emission_dim, num_timesteps, a, b):

        str = 'sim: covinfl\n'
        str += '\t{\n'
        str += '\t\temission_dim: {0},\n'.format(emission_dim)
        str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
        str += '\t\ta: {0},\n'.format(a)
        str += '\t\tb: {0}\n'.format(b)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'covinfl\{emission_dim:(.*),num_timesteps:(.*),a:(.*),b:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.emission_dim = int(m.group(1))
        self.num_timesteps = int(m.group(2))
        self.a = float(m.group(3))
        self.b = float(m.group(4))

    def get_dir(self):

        return os.path.join('covinfl', 'emission-dim_{0}_num-timesteps_{1}_a_{2}_b_{3}'.format(self.emission_dim, self.num_timesteps, self.a, self.b))


class MTT_Descriptor(SimulatorDescriptor):

    def __init__(self, str=None):

        self.state_dim = None
        self.emission_dim = None
        self.input_dim = 0
        self.num_timesteps = None
        self.a = None
        self.sigma2 = None
        self.name = 'mtt'

        try:

            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'mtt\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(self.state_dim)
        str += '\t\temission_dim: {0},\n'.format(self.emission_dim)

        if self.a is None:

            str += '\t\tnum_timesteps: {0}\n'.format(self.num_timesteps)

        else:

            str += '\t\tnum_timesteps: {0},\n'.format(self.num_timesteps)
            str += '\t\ta: {0},\n'.format(self.a)
            str += '\t\tsigma2: {0}\n'.format(self.sigma2)

        str += '\t}'

        return str

    def create_desc(self, state_dim, emission_dim, num_timesteps, a=None, sigma2=None):
        """The optional a (turn parameter) and sigma2 (variance of the observation noise) replace the defaults of simulators/mtt.py (a = 0.05, sigma2 = 2.5)."""

        str = 'sim: mtt\n'
        str += '\t{\n'
        str += '\t\tstate_dim: {0},\n'.format(state_dim)
        str += '\t\temission_dim: {0},\n'.format(emission_dim)

        if a is None:

            str += '\t\tnum_timesteps: {0}\n'.format(num_timesteps)

        else:

            str += '\t\tnum_timesteps: {0},\n'.format(num_timesteps)
            str += '\t\ta: {0},\n'.format(a)
            str += '\t\tsigma2: {0}\n'.format(sigma2)

        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'mtt\{state_dim:([^,]*),emission_dim:([^,]*),num_timesteps:([^,]*)(?:,a:([^,]*),sigma2:([^,]*))?\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.state_dim = int(m.group(1))
        self.emission_dim = int(m.group(2))
        self.num_timesteps = int(m.group(3))
        self.a = None if m.group(4) is None else float(m.group(4))
        self.sigma2 = None if m.group(5) is None else float(m.group(5))

    def get_dir(self):

        name = 'state-dim_{0}_emission-dim_{1}_num-timesteps_{2}'.format(self.state_dim, self.emission_dim, self.num_timesteps)

        if self.a is not None:

            name += '_a_{0}_sigma2_{1}'.format(self.a, self.sigma2)

        return os.path.join('mtt', name)
    

class InferenceDescriptor:

    @staticmethod
    def get_descriptor(str=None):

        if re.match('lgsf', str):

            return LGSF_Descriptor(str)

        elif re.match('ugsf', str):

            return UGSF_Descriptor(str)
        
        elif re.match('lagsf', str):

            return LAGSF_Descriptor(str)
        
        elif re.match('uagsf', str):

            return UAGSF_Descriptor(str)
        
        elif re.match('bpf', str):

            return BPF_Descriptor(str)
        
        elif re.match('apf', str):

            return APF_Descriptor(str)

        else:

            raise ParseError(str)


class GSF_Descriptor(InferenceDescriptor):

    def get_id(self):

        raise NotImplementedError('abstract method')

    def get_dir(self):

        return os.path.join('gsf', self.get_id())


class LGSF_Descriptor(GSF_Descriptor):

    def __init__(self, str=None):

        self.n_comps = None
        self.n_iters = None
        self.name = 'lgsf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lgsf\n'
        str += '\t{\n'
        str += '\t\tn_comps: {0},\n'.format(self.n_comps)
        str += '\t\tn_iters: {0}\n'.format(self.n_iters)
        str += '\t}'

        return str
    
    def create_desc(self, n_comps, n_iters):

        str = 'inf: lgsf\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tn_comps: {0},\n'.format(n_comps)
        str += '\t\tn_iters: {0}\n'.format(n_iters)
        str += '\n'
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'lgsf\{n_comps:(.*),n_iters:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_comps = int(m.group(1))
        self.n_iters = int(m.group(2))

    def get_id(self, delim='_'):

        id = 'ncomps' + delim + str(self.n_comps)
        id += delim + 'iters' + delim + str(self.n_iters)

        return id
    
    def get_dir(self):

        return os.path.join('gsf','lgsf', self.get_id())


class UGSF_Descriptor(GSF_Descriptor):

    def __init__(self, str=None):

        self.n_comps = None
        self.n_iters = None
        self.alpha = None
        self.beta = None
        self.kappa = None
        self.name = 'ugsf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'ugsf\n'
        str += '\t{\n'
        str += '\t\tn_comps: {0},\n'.format(self.n_comps)
        str += '\t\tn_iters: {0},\n'.format(self.n_iters)
        str += '\t\talpha: {0},\n'.format(self.alpha)
        str += '\t\tbeta: {0},\n'.format(self.beta)
        str += '\t\tkappa: {0}\n'.format(self.kappa)
        str += '\t}'

        return str
    
    def create_desc(self, n_comps, n_iters, alpha, beta, kappa):

        str = 'inf: ugsf\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tn_comps: {0},\n'.format(n_comps)
        str += '\t\tn_iters: {0},\n'.format(n_iters)
        str += '\t\talpha: {0},\n'.format(alpha)
        str += '\t\tbeta: {0},\n'.format(beta)
        str += '\t\tkappa: {0}\n'.format(kappa)
        str += '\n'
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'ugsf\{n_comps:(.*),n_iters:(.*),alpha:(.*),beta:(.*),kappa:(.*)\}\Z', str)

        if m is None:
            raise ParseError(str)

        self.n_comps = int(m.group(1))
        self.n_iters = int(m.group(2))
        self.alpha = float(m.group(3))
        self.beta = float(m.group(4))
        self.kappa = float(m.group(5))

    def get_id(self, delim='_'):

        id = 'ncomps' + delim + str(self.n_comps)
        id += delim + 'iters' + delim + str(self.n_iters)
        id += delim + 'alpha' + delim + str(self.alpha)
        id += delim + 'beta' + delim + str(self.beta)
        id += delim + 'kappa' + delim + str(self.kappa)

        return id
    
    def get_dir(self):

        return os.path.join('gsf','ugsf', self.get_id())


class LAGSF_Descriptor():

    def __init__(self, str=None):

        self.M = None
        self.N = None
        self.L = None
        self.n_iters = None
        self.ac1 =  None
        self.ac2 = None
        self.r1 = None
        self.r2 = None
        self.name = 'lagsf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'lagsf\n'
        str += '\t{\n'
        str += '\t\tM: {0},\n'.format(self.M)
        str += '\t\tN: {0},\n'.format(self.N)
        str += '\t\tL: {0},\n'.format(self.L)
        str += '\t\tn_iters: {0},\n'.format(self.n_iters)
        str += '\t\tac1: {0},\n'.format(self.ac1)
        str += '\t\tr1: {0},\n'.format(self.r1)
        str += '\t\tac2: {0},\n'.format(self.ac2)
        str += '\t\tr2: {0}\n'.format(self.r2)
        str += '\t}'

        return str
    
    def create_desc(self, M, N, L, n_iters, ac1, r1, ac2, r2):

        str = 'inf: lagsf\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tM: {0},\n'.format(M)
        str += '\t\tN: {0},\n'.format(N)
        str += '\t\tL: {0},\n'.format(L)
        str += '\t\tn_iters: {0},\n'.format(n_iters)
        str += '\t\tac1: {0},\n'.format(ac1)
        str += '\t\tr1: {0},\n'.format(r1)
        str += '\t\tac2: {0},\n'.format(ac2)
        str += '\t\tr2: {0}\n'.format(r2)
        str += '\n'
        str += '\t}'

        return str
    
    def parse(self, s):

        s = util.misc.remove_whitespace(s)
        m = re.match(r'lagsf\{M:(.*),N:(.*),L:(.*),n_iters:(.*),ac1:(.*),r1:(.*),ac2:(.*),r2:(.*)\}\Z', s)

        if m is None:
            raise ParseError(str)

        self.M = int(m.group(1))
        self.N = int(m.group(2))
        self.L = int(m.group(3))
        self.n_iters = int(m.group(4))
        self.ac1 = m.group(5)
        self.r1 = float(m.group(6))
        self.ac2 = m.group(7)
        self.r2 = float(m.group(8))

    def get_id(self, delim='_'):

        id = 'M' + delim + str(self.M)
        id += delim + 'N' + delim + str(self.N)
        id += delim + 'L' + delim + str(self.L)
        id += delim + 'iters' + delim + str(self.n_iters)
        id += delim + 'ac1' + delim + str(self.ac1)
        id += delim + 'r1' + delim + str(self.r1)
        id += delim + 'ac2' + delim + str(self.ac2)
        id += delim + 'r2' + delim + str(self.r2)

        return id
    
    def get_dir(self):

        return os.path.join('agsf', 'lagsf', self.get_id())


class UAGSF_Descriptor():

    def __init__(self, str=None):

        self.M = None
        self.N = None
        self.L = None
        self.n_iters = None
        self.alpha = None
        self.beta = None
        self.kappa = None
        self.ac1 =  None
        self.r1 = None
        self.ac2 = None
        self.r2 = None
        self.name = 'uagsf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'uagsf\n'
        str += '\t{\n'
        str += '\t\tM: {0},\n'.format(self.M)
        str += '\t\tN: {0},\n'.format(self.N)
        str += '\t\tL: {0},\n'.format(self.L)
        str += '\t\tn_iters: {0},\n'.format(self.n_iters)
        str += '\t\talpha: {0},\n'.format(self.alpha)
        str += '\t\tbeta: {0},\n'.format(self.beta)
        str += '\t\tkappa: {0},\n'.format(self.kappa)
        str += '\t\tac1: {0},\n'.format(self.ac1)
        str += '\t\tr1: {0},\n'.format(self.r1)
        str += '\t\tac2: {0},\n'.format(self.ac2)
        str += '\t\tr2: {0}\n'.format(self.r2)
        str += '\t}'

        return str
    
    def create_desc(self, M, N, L, n_iters, alpha, beta, kappa, ac1, r1, ac2, r2):

        str = 'inf: uagsf\n'
        str += '\n'
        str += '\t{\n'
        str += '\n'
        str += '\t\tM: {0},\n'.format(M)
        str += '\t\tN: {0},\n'.format(N)
        str += '\t\tL: {0},\n'.format(L)
        str += '\t\tn_iters: {0},\n'.format(n_iters)
        str += '\t\talpha: {0},\n'.format(alpha)
        str += '\t\tbeta: {0},\n'.format(beta)
        str += '\t\tkappa: {0},\n'.format(kappa)
        str += '\t\tac1: {0},\n'.format(ac1)
        str += '\t\tr1: {0},\n'.format(r1)
        str += '\t\tac2: {0},\n'.format(ac2)
        str += '\t\tr2: {0}\n'.format(r2)
        str += '\n'
        str += '\t}'

        return str
    
    def parse(self, s):

        s = util.misc.remove_whitespace(s)
        m = re.match(r'uagsf\{M:(.*),N:(.*),L:(.*),n_iters:(.*),alpha:(.*),beta:(.*),kappa:(.*),ac1:(.*),r1:(.*),ac2:(.*),r2:(.*)\}\Z', s)

        if m is None:
            raise ParseError(str)

        self.M = int(m.group(1))
        self.N = int(m.group(2))
        self.L = int(m.group(3))
        self.n_iters = int(m.group(4))
        self.alpha = float(m.group(5))
        self.beta = float(m.group(6))
        self.kappa = float(m.group(7))
        self.ac1 = m.group(8)
        self.r1 = float(m.group(9))
        self.ac2 = m.group(10)
        self.r2 = float(m.group(11))

    def get_id(self, delim='_'):

        id = 'M' + delim + str(self.M)
        id += delim + 'N' + delim + str(self.N)
        id += delim + 'L' + delim + str(self.L)
        id += delim + 'iters' + delim + str(self.n_iters)
        id += delim + 'ac1' + delim + str(self.ac1)
        id += delim + 'r1' + delim + str(self.r1)
        id += delim + 'ac2' + delim + str(self.ac2)
        id += delim + 'r2' + delim + str(self.r2)

        return id
    
    def get_dir(self):

        return os.path.join('agsf', 'uagsf', self.get_id())
    

class BPF_Descriptor():

    def __init__(self, str=None):

        self.num_prt = None
        self.ess = None
        self.name = 'bpf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):   

        str = 'bpf\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(self.num_prt)
        str += '\t\tess: {0}\n'.format(self.ess)
        str += '\t}'

        return str
    
    def create_desc(self, num_prt, ess):

        str = 'inf: bpf\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(num_prt)
        str += '\t\tess: {0}\n'.format(ess)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'bpf\{num_prt:(.*),ess:(.*)\}\Z', str)

        self.num_prt = int(m.group(1))
        self.ess = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'numprt' + delim + str(self.num_prt)
        id += delim + 'ess' + delim + str(self.ess)

        return id
    
    def get_dir(self):

        return os.path.join('pf', 'bpf', self.get_id())
    

class APF_Descriptor():

    def __init__(self, str=None):

        self.num_prt = None
        self.ess = None
        self.name = 'apf'

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):   

        str = 'apf\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(self.num_prt)
        str += '\t\tess: {0}\n'.format(self.ess)
        str += '\t}'

        return str
    
    def create_desc(self, num_prt, ess):

        str = 'inf: apf\n'
        str += '\t{\n'
        str += '\t\tnum_prt: {0},\n'.format(num_prt)
        str += '\t\tess: {0}\n'.format(ess)
        str += '\t}'

        return str

    def parse(self, str):

        str = util.misc.remove_whitespace(str)
        m = re.match(r'apf\{num_prt:(.*),ess:(.*)\}\Z', str)

        self.num_prt = int(m.group(1))
        self.ess = float(m.group(2))

    def get_id(self, delim='_'):

        id = 'numprt' + delim + str(self.num_prt)
        id += delim + 'ess' + delim + str(self.ess)

        return id
    
    def get_dir(self):

        return os.path.join('pf', 'apf', self.get_id())


class ExperimentDescriptor:

    def __init__(self, str=None):

        self.sim = None
        self.inf = None
        self.str = str

        try:
    
            self.parse(str)

        except:

            pass

    def pprint(self):

        str = 'experiment\n'
        str += '{\n'
        str += '\tsim: {0},\n'.format(self.sim.pprint())
        str += '\n'
        str += '\tinf: {0}\n'.format(self.inf.pprint())
        str += '}\n'

        return str

    def create_desc(self, sim_desc, inf_desc):

        str = 'experiment\n\n'
        str += '{\n\n\t'
        str += sim_desc
        str += ',\n\n\t'
        str += inf_desc
        str += '\n}'
        str += '\n'

        return str

    def parse(self, str):
        '''
        Parses the exp_descr string into sim and inf fields.
        '''

        str = util.misc.remove_whitespace(str)
        m = re.match(r'experiment\{sim:(.*),inf:(.*)\}\Z', str)

        if m is None:

            raise ParseError(str)

        self.sim = SimulatorDescriptor.get_descriptor(m.group(1))
        self.inf = InferenceDescriptor.get_descriptor(m.group(2))

    def get_dir(self):

        return os.path.join(self.sim.get_dir(), self.inf.get_dir())


def parse(str):
    """
    Parses the string str, and returns a list of experiment descriptor objects described by the string.
    """

    str = util.misc.remove_whitespace(str)
    descs = []
    pattern = re.compile(r'experiment\{')
    match = pattern.search(str)

    while match:

        exp_str = match.group()
        left = 1
        right = 0
        i = match.end()

        # consume the string until curly brackets close
        while left > right:

            try:

                if str[i] == '{':

                    left += 1

                if str[i] == '}':

                    right += 1

            # if we have reached the end of the string, discard current match
            except IndexError:

                print('Experiment not compiled. End reached without bracket closing.')
                print('')

                return descs

            exp_str += str[i]
            i += 1

        try:

            desc = ExperimentDescriptor(exp_str)
            descs.append(desc)

        except ParseError as err:

            print('Experiment not compiled. Parse error in:')
            print(err)
            print('')

        str = str[i:]
        match = pattern.search(str)

    return descs
