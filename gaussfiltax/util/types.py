from typing import Optional, Union
from typing_extensions import Protocol # type: ignore
from jaxtyping import Array, Float # type: ignore
import jax.random as jr # type: ignore
 

PRNGKey = jr.PRNGKey

Scalar = Union[float, Float[Array, ""]] # python float or scalar jax device array with dtype float
