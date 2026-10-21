# This module is taken/adapted from the repository ([https://github.com/gpapamak/snl.git])
# Originally authored by George Papamakarios, under the MIT License

import numpy as np  # type: ignore 
import matplotlib.pyplot as plt  # type: ignore
import experiment_descriptor as ed


class NonExistentExperiment(Exception):
    """
    Exception to be thrown when the requested experiment doesn't exist.
    """

    def __init__(self, exp_desc):
        assert isinstance(exp_desc, ed.ExperimentDescriptor)
        self.exp_desc = exp_desc

    def __str__(self):
        return self.exp_desc.pprint()


class AlreadyExistingExperiment(Exception):
    """
    Exception to be thrown when the requested experiment already exists.
    """

    def __init__(self, exp_desc):
        assert isinstance(exp_desc, ed.ExperimentDescriptor)
        self.exp_desc = exp_desc

    def __str__(self):
        return self.exp_desc.pprint()


def get_simulator(sim_desc):
    """
    Given the description of a simulator, returns the simulator module.
    """

    if isinstance(sim_desc, ed.LGSSM_Descriptor):

        from simulators import lgssm as sim

    elif isinstance(sim_desc, ed.LMSV_Descriptor):

        from simulators import lmsv as sim

    elif isinstance(sim_desc, ed.COVINFL_Descriptor):

        from simulators import covinfl as sim

    elif isinstance(sim_desc, ed.MTT_Descriptor):

        from simulators import mtt as sim

    else:

        raise ValueError('unknown simulator')

    return sim


def get_root():

    return '/Users/kostastsampourakis/Desktop/code/Python/projects/gaussfiltax/data.nosync'


def find_epsilon(sims_loader, obs_xs, acc_rate, show_hist=True):
    """
    Finds an epsilon for rejection ABC that yeilds a particular acceptance rate.
    :param sims_loader: a simulations loader object
    :param obs_xs: the observed data
    :param acc_rate: acceptance rate
    :param show_hist: if True, show a histogram of the distances
    :return: epsilon value
    """

    n_sims = 10**6
    xs = None

    while True:

        try:
            _, xs = sims_loader.load(n_sims)
            break

        except RuntimeError:
            n_sims /= 2

    dist = np.sqrt(np.sum((xs - obs_xs) ** 2, axis=1))
    eps = np.percentile(dist, acc_rate * 100)

    if show_hist:
        fig, ax = plt.subplots(1, 1)
        ax.hist(dist, bins='auto', normed=True)
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.vlines(eps, 0, ax.get_ylim()[1], color='r')
        ax.set_xlabel('distances')
        plt.show()

    return eps


from collections import defaultdict
import re


def get_varying(fname):

    with open(fname) as f:
        content = f.read()

    blocks = re.findall(r"experiment\s*{(.*?)}\s\}", content, re.DOTALL)

    configs = []
    for block in blocks:
        params = dict(re.findall(r"(\w+):\s*([\w.]+)", block))
        configs.append(params)

    varying_keys = defaultdict(set)
    for cfg in configs:
        for k, v in cfg.items():
            varying_keys[k].add(v)

    varying = {k: sorted(vs, key=lambda x: float(x)) for k, vs in varying_keys.items() if len(vs) > 1}

    for k in varying:
        varying[k] = list(map(int, varying[k]))

    variable_names = list(varying.keys())

    return varying, variable_names
