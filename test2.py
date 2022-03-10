from utils import Utils as u
import numpy as np

mean1 = np.array([1,1,1])
mean2 = np.array([3,7,2])
mean_mat = np.array([mean1, mean2])
w1 = 0.01
w2 = 0.99
weights_vec = np.array([w1,w2])
cov1 = np.eye(3)
cov2 = 10 * np.eye(3)
cov_tens = np.array([cov1, cov2])
m, P = u.collapse(mean_mat, cov_tens, weights_vec)