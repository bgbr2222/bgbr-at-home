# A Gaussian Mixture Model (GMM) is a convex some of Gaussian Distributions
# GMM(x) = \sum_i pi[i]*Gaussian(x, mean[i], cov[i])

# This class assumes data and distributions are 2D

import numpy as np
import scipy


class GMM:

    def __init__(self, pi=None, mean=None, cov=None):
        # create a GMM with arguments
        if (not pi) or (not mean) or (not cov):
            # create an empty GMM
            self.num_terms = 0
            self.pi = []
            self.mean = []
            self.cov = []
        else:
            assert (len(pi) == len(mean) == len(cov))

            self.num_terms = len(pi)    # number of components in GMM
            self.pi = pi                # list of weights
            self.mean = mean            # list of means
            self.cov = cov              # list of covariance matrices

    def __str__(self):
        # provides a nice way to output the details of our GMM
        str = f"<GMM @{id(self)} with {self.num_terms} components>\n"
        for k in range(self.num_terms):
            str += f"\t{k}: pi={self.pi[k]
                                }, mean={self.mean[k]}, cov={self.cov[k]}\n"

        return str

    def addTerm(self, new_pi, new_mean, new_cov):
        # adds a new term to the GMM and updates weights

        self.num_terms += 1
        # normalize existing weights so that they all sum to 1
        self.pi = [pi*(1-new_pi) for pi in self.pi]
        self.pi.append(new_pi)

        self.mean.append(new_mean)
        self.cov.append(new_cov)

    def evaluate(self, xy):
        # evaluate the GMM at the given points
        # xy is an N x 2 numpy array
        # returns an N x 1 array of values

        f = sum(self.pi[j]*scipy.stats.multivariate_normal.pdf(xy,
                self.mean[j], self.cov[j]) for j in range(0, self.num_terms))
        return f

    def sample(self, num_samples):
        # draws samples from the GMM
        # num samples is an integer
        # returns an num_samples x 2 numpy array of samples

        # sample from categorical distribution
        comp_samples = np.random.choice(
            range(self.num_terms), size=num_samples, p=self.pi)
        # sample from components
        xy = [scipy.stats.multivariate_normal.rvs(
            self.mean[comp_samples[k]], self.cov[comp_samples[k]], 1) for k in range(num_samples)]
        xy = np.vstack(xy)
        return xy

    def greedyEMFit(self, xy, max_components=20, max_iter=999):

        num_samples = xy.shape[0]

        # reset the GMM
        self.num_terms = 0
        self.pi = []
        self.mean = []
        self.cov = []

        # initialize the new GMM
        m = np.mean(xy, axis=0)
        S = np.cov(xy[:, 0], xy[:, 1], ddof=0)

        d = 2   # dimension of data
        beta = 0.5*max(scipy.linalg.svdvals(S))
        sigma = beta*(4/((d+2)*num_samples))**(1/(d+4))

        # define kernel matrix
        kij = np.array([[np.linalg.norm(pos1-pos2)**2 for pos2 in xy]
                        for pos1 in xy])
        K = (2*np.pi*sigma**2)**(-d/2)*np.exp((-0.5/sigma**2)*kij)

        self.num_terms = 1
        self.pi = [1.0]
        self.mean = [m]
        self.cov = [S]

        iter_count = 0
        break_flag = False
        while iter_count < max_iter:
            iter_count += 1

            f_k = self.evaluate(xy)
            #L_k = np.sum(np.log(f_k), axis=0)
            L_k = self.__EM_step(xy)           

            # search over xy for new mean
            def delta(j): return (f_k - K[:, j])/(f_k + K[:, j])
            n_idx = np.argmax([np.sum(np.log((f_k + K[:, j])/2), axis=0) +
                               0.5*np.sum(delta(j), axis=0)**2/np.dot(delta(j), delta(j)) for j in range(0, num_samples)])

            # initialize new component
            m_new = xy[n_idx, :]
            S_new = sigma**2*np.identity(2)
            phi_new = scipy.stats.multivariate_normal.pdf(xy, m_new, S_new)
            delta_new = (f_k - phi_new)/(f_k + phi_new)
            a = 0.5 - 0.5*np.sum(delta_new, axis=0)**2 / \
                np.dot(delta_new, delta_new)
            # check bounds of weight a
            if not (0 < a < 1):
                if self.num_terms == 1:
                    a = 0.5
                else:
                    a = 2/(self.num_terms+1)

            break_flag = self.__partial_EM_step(
                xy, f_k, L_k, a, m_new, S_new, phi_new)
            
            if break_flag:
                break

            if self.num_terms >= max_components:
                break

    def __partial_EM_step(self, xy, f_k, L_k, a, m_new, S_new, phi_new, conv_tol=1e-6):

        num_samples = xy.shape[0]

        L_k1 = np.sum(np.log((1-a)*f_k + a*phi_new), axis=0)
        L_k1_prev = 999

        while not (abs(L_k1/L_k1_prev - 1) < conv_tol):

            P_k_xi = a*phi_new/((1-a)*f_k + a*phi_new)
            SP_k_xi = np.sum(P_k_xi, axis=0)
            a = 1/num_samples*SP_k_xi
            m_new = np.dot(P_k_xi, xy)/SP_k_xi
            S_new = np.dot(P_k_xi, np.stack(
                [np.outer(row, row.T) for row in (xy-m_new)], axis=1))/SP_k_xi

            try: 
                phi_new = scipy.stats.multivariate_normal.pdf(xy, m_new, S_new)
            except:
                print('*** New covariance became singular')
                return True

            L_k1_prev = L_k1
            L_k1 = np.sum(np.log((1-a)*f_k + a*phi_new), axis=0)

        if (L_k1 >= L_k):
            self.addTerm(a, m_new, S_new)
            return False
        else:
            return True

    def __EM_step(self, xy, conv_tol=1e-6):

        num_samples = xy.shape[0]

        f_k = self.evaluate(xy)
        L_k = np.sum(np.log(f_k), axis=0)
        L_k_prev = 999

        P_xi = [np.empty(num_samples) for _ in range(0, self.num_terms)]

        while not (abs(L_k/L_k_prev - 1) < conv_tol):

            for j in range(0, self.num_terms):
                P_xi[j] = self.pi[j] * \
                scipy.stats.multivariate_normal.pdf(xy, self.mean[j], self.cov[j])/f_k
                S_Pxij = np.sum(P_xi[j], axis=0)
                self.pi[j] = 1/num_samples*S_Pxij
                self.mean[j] = np.dot(P_xi[j], xy)/S_Pxij
                self.cov[j] = np.dot(P_xi[j], np.stack([np.outer(row, row.T)
                                                        for row in (xy-self.mean[j])], axis=1))/S_Pxij

            f_k = self.evaluate(xy)
            L_k_prev = L_k
            L_k = np.sum(np.log(f_k), axis=0)

        return L_k

###

def randomGMM(num_components, var):
    # num_components is an integer, var is a float controlling the 'spread' of the covariance (suggested valueas are between 0 and 1)
    # returns a 'random' GMM with self.num_terms = num_components

    # create 'random' component weights that sum to 1
    pi = np.random.dirichlet(np.ones(num_components), size=1)[0].tolist()

    mean = []
    cov = []

    for i in range(num_components):
        # generate a random covariance matrix, via random orthogonal and positive diagonal matrices
        O = scipy.stats.ortho_group.rvs(2)
        D = np.diag(np.abs(np.random.normal(0, var, 2)))
        S = O@D@(O.T)

        # generate a random mean within the interval [-1,1]x[-1,1]
        mu = np.random.rand(2)*2-1

        mean.append(mu)
        cov.append(S)

    rGMM = GMM(pi, mean, cov)

    return rGMM

