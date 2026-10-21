import jax.numpy as jnp # type: ignore
from typing import List
import tensorflow_probability.substrates.jax.distributions as tfd # type: ignore
import tensorflow_probability.substrates.jax.bijectors as tfb # type: ignore
from util.bijectors import VecToCorrMat, RealToPSDBijector
from typing import Optional, runtime_checkable
from typing_extensions import Protocol # type: ignore
import abc
from jax.tree_util import register_pytree_node_class # type: ignore

@runtime_checkable
class ParameterSet(Protocol):
    """A :class:`NamedTuple` with parameters stored as :class:`jax.DeviceArray` in the leaf nodes.

    """
    pass

@runtime_checkable
class PropertySet(Protocol):
    """A matching :class:`NamedTuple` with :class:`ParameterProperties` stored in the leaf nodes.

    """
    pass


@register_pytree_node_class
class ParameterProperties:
    """A PyTree containing parameter metadata (properties).

    Note: the properties are stored in the aux_data of this PyTree so that
    changes will trigger recompilation of functions that rely on them.

    Args:
        trainable (bool): flag specifying whether or not to fit this parameter is adjustable.
        constrainer (Optional tfb.Bijector): bijector mapping to constrained form.

    """
    def __init__(self,
                 trainable: bool = True,
                 constrainer: Optional[tfb.Bijector] = None) -> None:

        self.trainable = trainable
        self.constrainer = constrainer

    def tree_flatten(self):

        return (), (self.trainable, self.constrainer)

    @classmethod
    def tree_unflatten(cls, aux_data, children):

        return cls(*aux_data)

    def __repr__(self):

        return f"ParameterProperties(trainable={self.trainable}, constrainer={self.constrainer})"


class ParamNode():
        
        def __init__(self, 
                    value: jnp.ndarray,
                    is_constrained: bool=False):
            
            self.value = value
            self.is_constrained = is_constrained

        def __str__(self):

            return f"value={self.value}, is_constrained={self.is_constrained}"


class ParamField():

    def __init__(self, 
                 values_list: List[jnp.ndarray],
                 param_names: List[str], 
                 is_constrained_list: List[bool]):

        for i, value in enumerate(values_list):

            setattr(self, param_names[i], ParamNode(value, is_constrained_list[i]))

    def __str__(self):

        str = ''

        for key in self.__dict__:

            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]}), "

        return str[:-2]
    

class Field():

    def __init__(self, 
                 values_list: List[jnp.ndarray],
                 param_names: List[str]=None):


        for i, value in enumerate(values_list):

            if param_names is not None:

                setattr(self, param_names[i], value)

            else:

                setattr(self, f"param_{i}", value)

    def __str__(self):

        str = ''

        for key in self.__dict__:

            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]}), "

        return str[:-2]


class ParamSSM():

    def __init__(self, 
                 initial: ParamField, 
                 dynamics: ParamField, 
                 emissions: ParamField):

        self.initial = initial
        self.dynamics = dynamics
        self.emissions = emissions

    def __str__(self):

        str = f'{self.__class__.__name__}('

        for key in ['initial', 'dynamics', 'emissions']:

            str += f"{key}={getattr(self, key).__class__.__name__}({self.__dict__[key]})"

        return str[:-2]

    def pprint(self):

        str = f'{self.__class__.__name__}(\n'

        for key in ['initial', 'dynamics', 'emissions']:

            str += f"\t {key}={getattr(self, key).__class__.__name__}(\n"

            for subkey in getattr(self, key).__dict__:

                str += f"\t \t{subkey}({getattr(getattr(self, key), subkey)}), \n"

            str += '\t\t ) \n'

        str += '\t )'

        return str

    def to_unconstrained(self, props):

        for field_name in ['initial', 'dynamics', 'emissions']:

            field = getattr(self, field_name)
            props_field = getattr(props, field_name)

            for subfield_name in field.__dict__:

                subfield = getattr(field, subfield_name)
                props_subfield = getattr(props_field, subfield_name)
                constrainer = props_subfield.props.constrainer

                if constrainer is not None:

                    setattr(subfield, 'is_constrained', False)

                    if isinstance(constrainer, tfb.bijector._AutoCompositeTensorBijectorMeta):

                        setattr(subfield, 'value', constrainer().inverse(subfield.value))

                    elif isinstance(constrainer, tfb.Exp) or isinstance(constrainer, VecToCorrMat):

                        setattr(subfield, 'value', constrainer.inverse(subfield.value))

                    else: 
                        
                        raise ValueError(f"Unknown constrainer type: {type(constrainer)}")

    def from_unconstrained(self, props):

        for field_name in ['initial', 'dynamics', 'emissions']:

            field = getattr(self, field_name)
            props_field = getattr(props, field_name)

            for subfield_name in field.__dict__:

                subfield = getattr(field, subfield_name)
                props_subfield = getattr(props_field, subfield_name)

                if not subfield.is_constrained:

                    constrainer = props_subfield.props.constrainer
                    setattr(subfield, 'is_constrained', True)

                    if constrainer is not None:

                        if isinstance(constrainer, tfb.Exp) or isinstance(constrainer, VecToCorrMat):

                            setattr(subfield, 'value', constrainer.forward(subfield.value))

                        else:

                            setattr(subfield, 'value', constrainer()(subfield.value))

    def _get_names(self):

        names_tree = []

        for field_name in ['initial', 'dynamics', 'emissions']:

            field = getattr(self, field_name)
            names_list = []

            for subfield_name in field.__dict__:

                names_list.append(subfield_name)

            names_tree.append(names_list)

        return names_tree
    
    def _is_constrained_tree(self):

        is_constrained_tree = []

        for field_name in ['initial', 'dynamics', 'emissions']:

            field = getattr(self, field_name)
            is_constrained_list = []

            for subfield_name in field.__dict__:

                is_constrained_list.append(getattr(getattr(field, subfield_name), 'is_constrained'))

            is_constrained_tree.append(is_constrained_list)

        return is_constrained_tree