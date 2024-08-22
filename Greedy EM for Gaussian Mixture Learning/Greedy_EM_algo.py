import matplotlib.pyplot as plt
import numpy as np
import scipy


# def xy_location_distribution(t, s, r, n_samples):
#     # this function returns samples (in Cartesian coordinates (km, km)) from the pdf 
#     # of an object based on the bistatic sonar equation, see eqn (5) of
#     # https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=4939494ad1d53bd31531254faa2fd7285e6bd8cc

#     # parameters
#     mu_c_s = 1.5                # mean speed of sound in water in km/s
#     sigma_c_s = 0.002           # var speed of sound in water in km/s
#     sigma_L = 0.02              # var of location error in km
#     sigma_phi = np.radians(5)   # var of azimuth in rad
#     sigma_tau = 0.01            # var of time in s
#     sigma_theta = np.radians(2) # var of heading in rad

#     c_s = scipy.stats.norm.rvs(loc=mu_c_s, scale=sigma_c_s, size=n_samples) 

#     phi = np.arctan2((t[0]-r[0]),(t[1]-r[1])) + \
#         scipy.stats.norm.rvs(loc=0, scale=sigma_phi, size=n_samples) - \
#             scipy.stats.norm.rvs(loc=0, scale=sigma_theta, size=n_samples)  # azimuth in rad
#     tau = (np.linalg.norm(t-s)+np.linalg.norm(t-r))/c_s + \
#         scipy.stats.norm.rvs(loc=0, scale=sigma_tau, size=n_samples)        # arrival time in s
#     #print(phi)
#     #print(tau)

#     BL = s - r + scipy.stats.norm.rvs(loc=0, scale=2*sigma_L, size=(n_samples,2))
#     alpha = np.arctan2(BL[:,0],BL[:,1]) - phi
#     delta = np.sqrt(BL[:,0]**2+BL[:,1]**2)
#     #print(BL)
#     #print(alpha)
#     #print(delta)
#     range = ((tau*c_s)**2-delta**2)/(2*(tau*c_s-delta*np.cos(alpha)))
#     #print(range)

#     x = np.sin(phi)*range + r[0]
#     y = np.cos(phi)*range + r[1]
#     #print(x)
#     #print(y)

#     return (x,y) #np.column_stack((x,y))

def partial_EM(xy, num_samples, f_k, a, m_new, S_new, phi_new):

    L_k1 = np.sum(np.log((1-a)*f_k + a*phi_new), axis=0)
    L_k1_prev = 999

    while not (abs(L_k1/L_k1_prev - 1) < 1e-6):

        P_k_xi = a*phi_new/((1-a)*f_k + a*phi_new)
        SP_k_xi = np.sum(P_k_xi, axis=0)
        a = 1/num_samples*SP_k_xi
        m_new = np.dot(P_k_xi, xy)/SP_k_xi
        S_new = np.dot(P_k_xi, np.stack([np.outer(row, row.T) for row in (xy-m_new)], axis=1))/SP_k_xi

        phi_new = scipy.stats.multivariate_normal.pdf(xy, m_new, S_new)
        L_k1_prev = L_k1
        L_k1 = np.sum(np.log((1-a)*f_k + a*phi_new), axis=0)

    return (L_k1, a, m_new, S_new)

def EM(xy, num_samples, f_k, k, pi, m, S):

    L_k = np.sum(np.log(f_k), axis=0)
    L_k_prev = 999

    P_xi = [np.empty(num_samples) for _ in range(0,k)]

    while not (abs(L_k/L_k_prev - 1) < 1e-6):

        # EM steps (4-7)
        for j in range(0,k):
            P_xi[j] = pi[j]*scipy.stats.multivariate_normal.pdf(xy, m[j], S[j])/f_k             
            S_Pxij = np.sum(P_xi[j], axis=0)
            pi[j] = 1/num_samples*S_Pxij
            m[j] = np.dot(P_xi[j], xy)/S_Pxij #np.sum(P_xi[j]*xy.T, axis=1)/np.sum(P_xi[j], axis=0)
            S[j] = np.dot(P_xi[j], np.stack([np.outer(row, row.T) for row in (xy-m[j])], axis=1))/S_Pxij

        f_k = sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy, m[j], S[j]) for j in range(0,k))
        L_k_prev = L_k
        L_k = np.sum(np.log(f_k), axis=0)

    return (L_k, pi, m, S)

def greedy_EM(x, y, num_samples):

    # initialize
    xy = np.column_stack((x,y))

    m = np.mean(xy, axis=0)
    S = np.cov(x, y, ddof=0)

    d = 2   # dimension of data
    beta = 0.5*max(scipy.linalg.svdvals(S))
    sigma = beta*(4/((d+2)*num_samples))**(1/(d+4))

    # kernel matrix
    kij = np.array([[np.linalg.norm(pos1-pos2)**2 for pos2 in xy] for pos1 in xy])
    K = (2*np.pi*sigma**2)**(-d/2)*np.exp((-0.5/sigma**2)*kij)

    k = 1       # number of components in mixture
    pi = [1]    # weights
    m = [m]
    S = [S]

    while True:

        f_k = sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy, m[j], S[j]) for j in range(0,k))

        (L_k, pi, m, S) = EM(xy, num_samples, f_k, k, pi, m, S)

        # search over xy for new mean 
        delta = lambda j : (f_k - K[:,j])/(f_k + K[:,j])
        n_idx = np.argmax([np.sum(np.log((f_k + K[:,j])/2), axis=0) + \
        0.5*np.sum(delta(j), axis=0)**2/np.dot(delta(j), delta(j)) for j in range(0, num_samples)])

        # initialize new component
        m_new = xy[n_idx,:]
        S_new = sigma**2*np.identity(2)
        phi_new = scipy.stats.multivariate_normal.pdf(xy, m_new, S_new)    
        delta_new = (f_k - phi_new)/(f_k + phi_new)
        a = 0.5 - 0.5*np.sum(delta_new, axis=0)**2/np.dot(delta_new, delta_new)
        # check bounds of weight a
        if not (0 < a < 1):
            if k == 1:
                a = 0.5
            else:
                a = 2/(k+1)

        (L_k1, a, m_new, S_new) = partial_EM(xy, num_samples, f_k, a, m_new, S_new, phi_new)

        if (L_k1 > L_k):
            # add new component
            k += 1
            m.append(m_new)
            S.append(S_new)
            pi = [pi_j*(1-a) for pi_j in pi]
            pi.append(a)
        else:
            break

    return (k, pi, m, S)

def get_sample(num_samples, num_clusters):

    x = np.empty(0)
    y = np.empty(0)

    pi = np.random.dirichlet(np.ones(num_clusters), size=1)[0].tolist()   
    #print(pi)
    #print(pre_weights)
    #print(sum(pre_weights))
    #pi = []
    m = []
    S = []

    smp_per = [int(pii*num_samples) for pii in pi]
    print(smp_per)
    #print(sum(smp_per))
    # normalize
    rem = num_samples - sum(smp_per)
    smp_per[0] += rem
    #print(sum(smp_per))
    #rem = num_samples % num_clusters

    
    for i in range(0, num_clusters):

        O = scipy.stats.ortho_group.rvs(2)
        D = np.diag(np.abs(np.random.normal(0,0.1,2)))
        
        mu = np.random.rand(2)*2-1
        G = O@D@(O.T)

        # print(mu.shape)
        # print(G.shape)
        # print(tot)
        xy = scipy.stats.multivariate_normal.rvs(mu, G, smp_per[i]) 
        #print(x.shape)
        #print(xy[:,0].shape)
        print(xy)

        if smp_per[i] == 1:
            x = np.concatenate((x, [xy[0]]), axis=0)
            y = np.concatenate((y, [xy[1]]), axis=0)
        elif smp_per[i] > 1:
            x = np.concatenate((x, xy[:,0]), axis=0)
            y = np.concatenate((y, xy[:,1]), axis=0)

        m.append(mu)
        S.append(G)

    #print(x)
    #print(y)

    return (x, y, pi , m, S)

def main():
    
    np.random.seed(11) # fix seed for test and debbug

    #t = np.array([2, 2])    # target location in Cartesian coords in (km, km)
    #s = np.array([-1, 0])   # source location in Cartesian coords in (km, km)
    #r = np.array([1, 0])    # receiver location in Cartesian coords in (km, km)

    num_samples = 200
    #(x, y) = xy_location_distribution(t, s, r, num_samples)

    num_clusters = 4

    (x, y, pia, ma, Sa) = get_sample(num_samples, num_clusters)
    
    print(num_clusters)
    print(pia)
    print(ma)
    print(Sa)

    (k, pi, m, S) = greedy_EM(x, y, num_samples)

    print(k)
    print(pi)
    print(m)
    print(S)

    ### plotting

    fig, ax = plt.subplots()

    #xx = np.linspace(-4, 4, 100)    
    X, Y = np.mgrid[-2:2:0.01, -2:2:0.01]
    xy_grid = np.dstack((X,Y))

    #Z = scipy.stats.multivariate_normal.pdf(xy_grid, m[0], S[0]) #sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy_grid, m[j], S[j]) for j in range(0,k))
    #Z = sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy_grid, m[j], S[j]) for j in range(0,k))
    Z = sum(pi[j]*scipy.stats.multivariate_normal.pdf(xy_grid, m[j], S[j])/scipy.stats.multivariate_normal.pdf(m[j], m[j], S[j]) for j in range(0,k))
    #print(np.max(Z))
    cs = ax.contour(X, Y, Z, cmap=plt.cm.YlOrRd)
    #cbar = fig.colorbar(cs)

    Ztrue = sum(pia[j]*scipy.stats.multivariate_normal.pdf(xy_grid, ma[j], Sa[j])/scipy.stats.multivariate_normal.pdf(ma[j], ma[j], Sa[j]) for j in range(0,num_clusters))
    cstrue = ax.contour(X, Y, Ztrue, cmap=plt.cm.PuBuGn)

    ax.grid()
    ax.scatter(x, y, c='b',marker='.', label='Target Samples')
    # ax.scatter(t[0], t[1], c='k', marker='x', label='Actual Target')
    # ax.scatter(s[0], s[1], c='k', marker='D', label='Source')
    # ax.scatter(r[0], r[1], c='k', marker='s', label='Receiver')
    # #ax.set_aspect('equal')
    # ax.set_xlabel("x (km)")
    # ax.set_ylabel("y (km)")
    #legend = ax.legend(loc='lower center')



    plt.axis('square')
    plt.show()
    

if __name__ == '__main__':
    main()


