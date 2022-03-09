import numpy as np

class Utils:
    
    def __init__(self):
        pass
    
    def collapse(mean_mat, covariance_tens, weight_vec):
        M,dx = np.shape(mean_mat)
        mean_out = np.matmul(weight_vec, mean_mat)
        cov_out = np.zeros([dx,dx])
        for m in range(M):
            diff = mean_mat[m] - mean_out
            cov_out = cov_out + weight_vec[m]*(covariance_tens[m] + 
                                               np.tensordot(diff, diff, axes =0))
        return mean_out, cov_out
            
        