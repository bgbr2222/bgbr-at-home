import matplotlib.pyplot as plt
import numpy as np
import scipy

from GaussianMixtureModel import *

def main():

    np.random.seed(42)  # fix seed for test and debbug

    # random sample parameters
    num_clusters = 4
    num_samples = 100
    cluster_size = 0.02

    randGMM = randomGMM(num_clusters, cluster_size)
    print(randGMM)
    xy_sample = randGMM.sample(num_samples)

    fitGMM = GMM()
    fitGMM.greedyEMFit(xy_sample)
    print(fitGMM)

    # plotting

    fig, ax = plt.subplots()

    X, Y = np.mgrid[-2:2:0.01, -2:2:0.01]
    xy_grid = np.dstack((X, Y))

    # Z = sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy_grid, m[j], S[j])/scipy.stats.multivariate_normal.pdf(m[j], m[j], S[j]) for j in range(0,k))
    # Z = fitGMM.evaluate(xy_grid)
    Z = sum(fitGMM.pi[j]*scipy.stats.multivariate_normal.pdf(xy_grid, fitGMM.mean[j], fitGMM.cov[j]) /
            scipy.stats.multivariate_normal.pdf(fitGMM.mean[j], fitGMM.mean[j], fitGMM.cov[j]) for j in range(0, fitGMM.num_terms))
    cs = ax.contour(X, Y, Z, cmap=plt.cm.YlOrRd)
    # cbar = fig.colorbar(cs)

    # Ztrue = sum(pia[j]*scipy.stats.multivariate_normal.pdf(xy_grid, ma[j], Sa[j])/scipy.stats.multivariate_normal.pdf(ma[j], ma[j], Sa[j]) for j in range(0,num_clusters))
    # cstrue = ax.contour(X, Y, Ztrue, cmap=plt.cm.PuBuGn)

    ax.grid()
    # colors = plt.cm.rainbow(np.linspace(0, 1, fitGMM.num_terms))
    # print(colors)
    # cc = np.zeros(len(xy_sample[:,0]))
    # for i, row in enumerate(xy_sample):
    #     cc[i] = colors[0]


    ax.scatter(xy_sample[:, 0], xy_sample[:, 1], color='b',
               marker='.', label='Target Samples')
    # ax.scatter(t[0], t[1], c='k', marker='x', label='Actual Target')
    # ax.scatter(s[0], s[1], c='k', marker='D', label='Source')
    # ax.scatter(r[0], r[1], c='k', marker='s', label='Receiver')
    # #ax.set_aspect('equal')
    # ax.set_xlabel("x (km)")
    # ax.set_ylabel("y (km)")
    # legend = ax.legend(loc='lower center')

    plt.axis('square')
    plt.show()


if __name__ == '__main__':
    main()
