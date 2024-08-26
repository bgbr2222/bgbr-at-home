import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn.cluster
import sklearn.mixture

from GaussianMixtureModel import *

# TODO add some comparison with scikit learn methods and update visualizations


def main():

    np.random.seed(42)  # fix seed for test and debbug

    max_terms = 20

    # random sample parameters
    num_clusters = 4
    num_samples = 100
    cluster_size = 0.02

    randGMM = randomGMM(num_clusters, cluster_size)
    # print(randGMM)
    xy_sample = randGMM.sample(num_samples)

    fitGMM = GMM()
    fitGMM.greedyEMFit(xy_sample, max_components=max_terms)
    #print(fitGMM)

    greedyGMM_cluster = np.array([np.argmin([scipy.spatial.distance.mahalanobis(
        xy, fitGMM.mean[k], np.linalg.inv(fitGMM.cov[k])) for k in range(fitGMM.num_terms)]) for xy in xy_sample])
    #print(greedyGMM_cluster)

    # comparisons

    KMeans_conv_tol = 0.5
    GM_conv_tol = 1e-2

    # sklearn KMeans
    skKM_num_terms = 1
    L_k_prev = 999
    # print(kmeans.labels_)
    while skKM_num_terms < max_terms:
        
        kmeans = sklearn.cluster.KMeans(
            n_clusters=skKM_num_terms, random_state=42, n_init="auto").fit(xy_sample)
        L_k = kmeans.inertia_
        # print(L_k)
        # print(abs(L_k/L_k_prev-1))
        if abs(L_k/L_k_prev-1) < KMeans_conv_tol:
            # print(skKM_num_terms)
            break
        L_k_prev = L_k
        skKM_num_terms += 1

    skKM_cluster = kmeans.labels_
    # print(skKM_cluster)

    # sklearn GM
    skGM_num_terms = 1
    L_k_prev = 999
    while skGM_num_terms < max_terms:
        
        skGM = sklearn.mixture.GaussianMixture(
            n_components=skGM_num_terms, random_state=42).fit(xy_sample)
        L_k = skGM.score(xy_sample)
        # print(L_k)
        # print(abs(L_k/L_k_prev-1))
        if abs(L_k/L_k_prev-1) < GM_conv_tol:
            # print(skGM_num_terms)
            break
        L_k_prev = L_k
        skGM_num_terms += 1

    skGM_cluster = skGM.predict(xy_sample)
    # print(skGM_cluster)

    # # plotting

    fig = plt.figure(figsize=(8, 3))

    colors = ['b', 'g', 'r', 'c', 'm', 'y']
    
    # GreedyEM GM
    ax = fig.add_subplot(1, 3, 1)
    ax.scatter(xy_sample[:, 0], xy_sample[:, 1], color=[colors[ind] for ind in greedyGMM_cluster],
                marker='.')  # , label='Target Samples')
    ax.set_title(f"GreedyEM GM Algorithm\n({fitGMM.num_terms} terms)")
    ax.grid()

    #skGM
    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(xy_sample[:, 0], xy_sample[:, 1], color=[colors[ind] for ind in skGM_cluster],
                marker='.')  # , label='Target Samples')
    ax.set_title(f"sklearn GM Algorithm\n({skGM_num_terms} terms)")
    ax.grid()

    #skKM
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(xy_sample[:, 0], xy_sample[:, 1], color=[colors[ind] for ind in skKM_cluster],
                marker='.')  # , label='Target Samples')
    ax.set_title(f"sklearn KMeans Algorithm\n({skKM_num_terms} terms)")
    ax.grid()

    plt.show()


if __name__ == '__main__':
    main()
