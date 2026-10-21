import tensorflow_probability.substrates.jax.bijectors as tfb # type: ignore
import jax.numpy as jnp # type: ignore
from jax import lax # type: ignore
from jax.scipy.linalg import expm, eigh, cholesky # type: ignore


# From https://www.tensorflow.org/probability/examples/
# TensorFlow_Probability_Case_Study_Covariance_Estimation

class PSDToRealBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
          
            tfb.Invert(tfb.FillTriangular()),
            tfb.TransformDiagonal(tfb.Invert(tfb.Exp())),
      
            tfb.Invert(tfb.CholeskyOuterProduct()),
        ]
      
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class RealToPSDBijector(tfb.Chain):

    def __init__(self,
                 validate_args=False,
                 validate_event_size=False,
                 parameters=None,
                 name=None):

        bijectors = [
  
            tfb.CholeskyOuterProduct(),
            tfb.TransformDiagonal(tfb.Exp()),
            tfb.FillTriangular(),
  
        ]
  
        super().__init__(bijectors, validate_args, validate_event_size, parameters, name)


class VecToLowerTriangularBijector(tfb.Bijector):
 
    def __init__(self, d, validate_args=False, name="LowerTriangularToVec"):
 
        self.d = d
        self.tril_i, self.tril_j = jnp.tril_indices(d, k=-1)
   
        super().__init__(
   
            forward_min_event_ndims=2,
            validate_args=validate_args,
            name=name
   
        )

    def inverse(self, x, name=None):
   
        return x[self.tril_i, self.tril_j]

    def forward(self, y, name=None):
   
        x = jnp.eye(self.d)
        x = x.at[self.tril_i, self.tril_j].set(y)
   
        return x


class VecToCorrMat(tfb.Bijector):

    def __init__(self, d, validate_args=False, name="VecToCorrMat"):

        self.d = d
        self.tril_i, self.tril_j = jnp.tril_indices(d, k=-1)
   
        super().__init__(

            forward_min_event_ndims=2,
            validate_args=validate_args,
            name=name

        )

    def inverse(self, chol):

        """
        Computes gamma(C) = vecl(log(C)) using eigendecomposition.
        """

        C = chol @ chol.T
        n = C.shape[0]
        assert self.d == n, f"Expected d={self.d}, but got {n}."
        eigvals, eigvecs = eigh(C)
        log_eigvals = jnp.log(eigvals)
        log_C = eigvecs @ jnp.diag(log_eigvals) @ eigvecs.T
        vecl_log_C = log_C[jnp.tril_indices(n, k=-1)]

        return vecl_log_C

    def corr_from_gamma(self, gamma, tol=1e-10, max_iter=1000):

        """
        Reconstruct correlation matrix C from gamma vector using fixed-point iteration.
        gamma: vector of length n(n-1)/2
        n: dimension of the desired correlation matrix
        """

        A = jnp.zeros((self.d, self.d))
        tril_indices = jnp.tril_indices(self.d, k=-1)
        A = A.at[tril_indices].set(gamma)
        A = A.at[(tril_indices[1], tril_indices[0])].set(gamma)  # Symmetrize
        x = jnp.zeros(self.d)

        def cond_fun(state):
            x, delta, iter_count = state
            return (jnp.linalg.norm(delta) >= tol) & (iter_count < max_iter)

        def body_fun(state):
            x, _, iter_count = state
            G = A + jnp.diag(x)
            C = expm(G)
            diag_C = jnp.diag(C)
            delta = jnp.log(diag_C)
            x = x - delta
            return x, delta, iter_count + 1

        x, _, _ = lax.while_loop(cond_fun, body_fun, (x, jnp.ones_like(x), 0))

        G = A + jnp.diag(x)

        return expm(G)
    
    def forward(self, gamma, tol=1e-10, max_iter=1000):
        """
        Maps gamma to the lower Cholesky factor L such that C = L @ L.T
        """

        C = self.corr_from_gamma(gamma, tol=tol, max_iter=max_iter)

        return cholesky(C, lower=True)


   
