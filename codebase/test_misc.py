import numpy as np
import utils
import matplotlib.pyplot as plt
import scipy.stats as stats
import math


old_mean = 0
old_cov = 1
new_cov = 0.001
num_comp = 100
xx = np.linspace(old_mean - 5*np.sqrt(old_cov), old_mean + 5*np.sqrt(old_cov), 1000)
sigma = math.sqrt(new_cov)
means = utils.split_by_sampling(np.array([old_mean]), np.array([old_cov]), np.array([new_cov]), num_comp)

sigma_points = utils.split_to_sigma_points(np.zeros(2), np.eye(2), 1, 2)
print(sigma_points)

mean_collapsed, cov_collapsed = utils.collapse(means, np.ones(num_comp)*new_cov, np.ones(num_comp)/num_comp)

fig3, axes3 = plt.subplots(1,1)
axes3.plot(xx, stats.norm.pdf(xx, old_mean, np.sqrt(old_cov)))
axes3.plot(xx, stats.norm.pdf(xx, mean_collapsed, np.sqrt(cov_collapsed[0])))
for mean in means:
    axes3.plot(xx, stats.norm.pdf(xx, mean, sigma) / num_comp)
#axes3.plot(xx, utils.gm(xx, means, sigma, num_comp))

plt.show()