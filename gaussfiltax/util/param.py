import jax.numpy as jnp # type: ignore
import jax.random as jr # type: ignore
from jax.tree_util import tree_map # type: ignore
from jax.flatten_util import ravel_pytree # type: ignore
from jax import jit # type: ignore
from typing import Union, Tuple, List
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import numpyro.distributions as npyrod # type: ignore
import tensorflow_probability.substrates.jax.bijectors as tfb # type: ignore
from functools import reduce, partial
from jax.tree_util import tree_map # type: ignore
from parameters import ParameterProperties, ParamField, Field, ParamSSM
import numpyro.distributions as dist # type: ignore

PRNGKey = jr.PRNGKey


def get_unravel_fn(params, props):

    list_trainable_params = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        props_field = getattr(props, field_name)
        sublist_trainable_params = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.props.trainable:
                sublist_trainable_params.append(subfield.value)
            else:
                sublist_trainable_params.append(jnp.array([]))
        list_trainable_params.append(sublist_trainable_params)
    _, unravel_fn = ravel_pytree(list_trainable_params)
    return unravel_fn


def to_train_array(params, props):
    '''
    Convert the parameters to a flat array of trainable parameters in unconstrained form.

    :param params: ParamSSM object with trainable parameters in unconstrained form
    :param props: properties of the paramters

    :return: flat array of concatenated trainable parameters in unconstrained form. If all parameters are
             in constrained form, returns an empty array.
    '''
    list_trainable_params = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        props_field = getattr(props, field_name)
        sublist_trainable_params = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.props.trainable:
                if subfield.is_constrained: 
                    constrainer = props_subfield.props.constrainer
                    if constrainer is not None:
                        val = constrainer().inverse(subfield.value)
                    else:
                        val = subfield.value
                    sublist_trainable_params.append(val)
                else:
                    sublist_trainable_params.append(subfield.value)
            else:
                sublist_trainable_params.append(jnp.array([]))
        list_trainable_params.append(sublist_trainable_params)
    train_array, _ = ravel_pytree(list_trainable_params)
    return train_array


def tree_from_params(params):
    tree = []
    for field_name in ['initial', 'dynamics', 'emissions']:
        field = getattr(params, field_name)
        subtree = []
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            subtree.append(subfield.value)
        tree.append(subtree)
    return tree


def params_from_tree(tree, names_tree, is_constrained_tree):
    initial = ParamField(tree[0], names_tree[0], is_constrained_tree[0])
    dynamics = ParamField(tree[1], names_tree[1], is_constrained_tree[1])
    emissions = ParamField(tree[2], names_tree[2], is_constrained_tree[2])
    return ParamSSM(initial, dynamics, emissions)


def join_trees(train_tree, untrain_tree, props):
    r"""Join two trees of parameters, one with trainable parameters and the other with untrainable parameters.
    The trainable parameter tree is the output of 
    """
    new_tree = []
    for i, field_name in enumerate(['initial', 'dynamics', 'emissions']):
        props_field = getattr(props, field_name)
        new_subtree = []
        for j, subfield_name in enumerate(props_field.__dict__):
            props_subfield = getattr(props_field, subfield_name)
            if props_subfield.props.trainable:
                new_subtree.append(train_tree[i][j])
            else:
                new_subtree.append(untrain_tree[i][j])
        new_tree.append(new_subtree)
    return new_tree


def sample_prior(
    key: PRNGKey, # type: ignore
    prior: ParamSSM, 
    num_samples: int=1):
    r"""Sample parameters from the prior distribution. When prior field is tfd.Distribution,
        sample num_samples values from it, and set the corresponding values of is_constrained to False
        (we consider that parameters are sampled always in unconstrained form). Otherwise set the value 
        equal to the provided value and is_constrained to True.
    """
    tree = []
    param_names = []
    is_constrained_tree = []

    for field_name in ['initial', 'dynamics', 'emissions']:

        field = getattr(prior, field_name)
        subtree = []
        param_subnames = []
        is_constrained_subtree = []

        for subfield_name in field.__dict__:

            subfield = getattr(field, subfield_name)

            if isinstance(subfield.prior, tfd.Distribution):

                key, subkey = jr.split(key)
                value = subfield.prior.sample(num_samples, subkey)
                is_constrained_subtree.append(False)

            elif isinstance(subfield.prior, npyrod.continuous.MatrixNormal):

                key, subkey = jr.split(key)
                value = subfield.prior.sample(subkey, (num_samples,))
                is_constrained_subtree.append(False)

            else:

                value = jnp.tile(subfield.prior, (num_samples, *tuple(1 for _ in range(subfield.prior.ndim))))
                is_constrained_subtree.append(True)

            subtree.append(value)
            param_subnames.append(subfield_name)

        is_constrained_tree.append(is_constrained_subtree)
        param_names.append(param_subnames)
        tree.append(subtree)
        
    return list(map(lambda i: params_from_tree(tree_map(lambda x: x[i], tree), param_names, is_constrained_tree), range(num_samples)))


def initialize(
        prior_fields: List[List[Union[jnp.ndarray, tfd.Distribution]]],
        param_names: List[List[str]],
        constrainers: List[List[Union[None, tfb.Bijector]]] = None
) -> Tuple[ParamSSM, ParamSSM]:
    r"""Initialize model parameters that are set to None, and their corresponding properties.

    Args:
        All arguments can either be Arrays or tfd.Distributions. The prior defaults to delta distributions if
        Arrays are provided, otherwise the prior is set to the provided distribution. Setting parameters to None
        results in delta priors at the default values defined in this function.

    Returns:

    """
    is_trainable = tree_map(lambda field: isinstance(field, tfd.Distribution) | isinstance(field, dist.MatrixNormal), prior_fields, is_leaf=lambda x: not isinstance(x, list))
    properties_tree = tree_map(lambda is_trainable, constrainer: ParameterProperties(is_trainable, constrainer), is_trainable, constrainers)
    props_prior_tree = tree_map(lambda props, prior:  Field([props, prior], ['props', 'prior']), properties_tree, prior_fields, is_leaf=lambda x: not isinstance(x, list))

    props = ParamSSM(
        initial = Field(
        props_prior_tree[0],
        param_names[0]),
        dynamics = Field(
            props_prior_tree[1],
            param_names[1]),
        emissions = Field(
            props_prior_tree[2],
            param_names[2])
            )

    return props


@partial(jit, static_argnums=(1,))
def log_prior(cond_params, props):
    r"""Compute the log prior of the parameters.

    :param params: parameters
    :param prior: prior distribution

    :return: log prior

    """
    log_prior = 0
    idx = 0
    for field_name in props.__dict__:
        
        field = getattr(props, field_name)
        
        for subfield_name in field.__dict__:
            
            subfield = getattr(field, subfield_name)
            
            if isinstance(subfield.prior, tfd.Distribution):

                try:
                    
                    shape = tuple(subfield.prior.event_shape)

                except:
                    
                    shape = subfield.prior.shape

                flat_shape = reduce(lambda x, y: x*y, shape)
                value = cond_params[idx:idx+flat_shape]
                log_prior += subfield.prior.log_prob(value)
                idx += flat_shape
    
    return log_prior


def jitter(key, params, props, jitter_scale=1e-2):
    r"""Compute the log prior of the parameters.

    :param params: parameters
    :param prior: prior distribution

    :return: log prior

    """
    for field_name in params.__dict__:
        field = getattr(params, field_name)
        for subfield_name in field.__dict__:
            subfield = getattr(field, subfield_name)
            subfield_props = getattr(getattr(props, field_name), subfield_name)
            if subfield_props.props.trainable:
                key, subkey = jr.split(key)
                shape = tuple(subfield.value.shape)
                noise = jitter_scale * jr.normal(subkey, shape)
                subfield.value += noise


def from_conditional(xparam, props, cond_params):
    '''
    Convert the parameters to a flat array of trainable parameters in constrained form.
    '''
    param_names = xparam._get_names()
    is_constrained_tree = xparam._is_constrained_tree()
    unravel_fn = get_unravel_fn(xparam, props)
    unravel = unravel_fn(cond_params)
    tree = tree_from_params(xparam)
    new_tree = join_trees(unravel, tree, props)
    params = params_from_tree(new_tree, param_names, is_constrained_tree)
    params.from_unconstrained(props)

    return params