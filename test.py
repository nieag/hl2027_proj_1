from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
import numpy as np

x, y = np.mgrid[-1:1:.01, -1:1:.01]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
print(rv)
plt.plot(np.ravel(rv.pdf(pos)))
plt.show()
    
