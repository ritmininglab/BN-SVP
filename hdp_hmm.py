#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from numpy.random import choice, normal, dirichlet, beta, gamma, multinomial, exponential, binomial

from scipy.cluster.vq import kmeans2
import copy
class StickyHDPHMM:
    def __init__(self, data, alpha = 1, kappa = 1, gma = 1, nu = 2, sigma_a = 2, sigma_b = 2, L = 10, kmeans_init = False):
        self.L = L
        self.alpha = alpha
        self.gma = gma
        self.data = data
        self.n, self.T, self.dim = self.data.shape
        
        num_instances = self.data.shape[0]*self.data.shape[1]
        self.state = choice(self.L, (len(data), self.T))
        self.kappa = kappa*num_instances
        std = 1
        self.mu = [[] for i in range(L)]
        self.sigma = [[] for i in range(L)]
        # Hyperparameters 
        self.nu = nu
        self.a = sigma_a
        self.b = sigma_b
        
        for i in range(L):
            idx = np.where(self.state==i)
            
            if len(self.data[idx])>1:
                cluster_data = self.data[idx].reshape(-1, self.dim)
                self.mu[i] = np.mean(cluster_data.reshape(-1, self.dim), axis = 0)
                self.sigma[i] = np.cov(cluster_data, rowvar = False)
            else:
                
                self.mu[i] =  np.random.multivariate_normal(np.zeros(self.dim), np.diag(np.repeat(np.sqrt(self.nu), self.dim)))
                self.sigma[i] = np.diag(1/gamma(np.repeat(self.a, self.dim), np.repeat(self.b, self.dim)))
        

        stickbreaking = self._gem(self.gma)
        self.beta = np.array([next(stickbreaking) for i in  range(L)])
        
        self.N = np.zeros((L, L))
       
        for i in range(self.n):
            for t in range(1, self.T):
                self.N[self.state[i, t-1], self.state[i, t]] = copy.deepcopy(self.N[self.state[i, t-1], self.state[i, t]]+1)
        self.M = np.zeros(self.N.shape)
        self.PI = (self.N.T/(np.sum(self.N, axis = 1)+1e-07)).T
        
        
    
    
    def _logphi(self, x, mu, sigma):
        if type(mu) is list:
            mu = np.array(mu)
            sigma = np.array(sigma)
        
            diff = np.reshape(x-mu, (self.L, self.dim))
            
            term1 = np.einsum('ij, ijj->ij', diff, sigma)
            term2 = np.einsum('ij, ij->i', term1, diff)
            
            term3 = 2*self.dim*np.pi*(np.linalg.det(sigma)+1e-06)
            
            
            loglikelihood = term2-term3
            return loglikelihood
        else:
           
            diff = np.reshape(x-mu, (1, self.dim))
           
            return np.dot(np.dot(diff, sigma), diff.T)-np.log(2*self.dim*np.pi*(np.linalg.det(sigma)+1e-06))
    
 
        
        
        
    
   
    def sampler(self):
        
        """
        Run blocked-Gibbs sampling
        
        """
        
        for i in range(self.n):
            messages = np.zeros((self.T, self.L))
            messages[-1, :] = 1
            for t in range(self.T-1, 0, -1):
                messages[t-1, :] = self.PI.dot(messages[t, :]*np.exp(self._logphi(self.data[i, t], self.mu, self.sigma)))
                messages[t-1, : ]/=np.max(messages[t-1, :])
       
          
            old_state = copy.deepcopy(self.state[i])
            for t in range(1, self.T):
                j = choice(self.L)
                k = copy.deepcopy(self.state[i, t])
               
                
                logprob_accept = (np.log(messages[t, k])-
                                  np.log(messages[t, j])+
                                  np.log(self.PI[self.state[i, t-1], k])-
                                  np.log(self.PI[self.state[i, t-1], j])+
                                  self._logphi(self.data[i, t-1], self.mu[k], self.sigma[k])-
                                  self._logphi(self.data[i, t-1], self.mu[j], self.sigma[j])
                )
                
                if old_state[t-1]!=self.state[i][t-1]:
                    if exponential(1)>logprob_accept:
                        self.N[old_state[t-1], k]-=1
                        self.N[self.state[i][t-1], j]+=1
                        self.state[i][t] = copy.deepcopy(j)
                    else:
                        self.N[old_state[t-1], k]-=1
                        self.N[self.state[i][t-1], k]+=1
                else:
                    if exponential(1)>logprob_accept:
                        self.state[i][t] = copy.deepcopy(j)
                        self.N[self.state[i][t-1], j]+=1
                        self.N[self.state[i][t-1], k] -=1
                    
           
            
        P = np.tile(self.beta, (self.L, 1))+self.n
        np.fill_diagonal(P, np.diag(P)+self.kappa)
        P = 1-self.n/P
        for i in range(self.L):
            for j in range(self.L):
                self.M[i, j] = binomial(self.M[i, j], P[i, j])
        
        w = np.array([binomial(self.M[i, i], 1/(1+self.beta[i])) for i in range(self.L)])
        m_bar = np.sum(self.M, axis = 0)-w
        self.beta = dirichlet(np.ones(self.L)*(self.gma/self.L))
        
        self.PI = np.tile(self.alpha*self.beta, (self.L, 1))+self.N
        np.fill_diagonal(self.PI, np.diag(self.PI)+self.kappa)
        
        for i in range(self.L):
            
            self.PI[i, :] = dirichlet(self.PI[i, :])
            idx = np.where(self.state==i)
            cluster_data = self.data[idx].reshape(-1, self.dim)
            nc = len(cluster_data)
            if nc>1:
                xmean = np.mean(cluster_data, axis = 0)
                self.mu[i] = xmean/(self.nu/nc+1)
                
                self.sigma[i] = (2*self.b+(nc-1)*np.cov(cluster_data, rowvar = False)+nc*xmean**2/(self.nu+nc))/(2*self.a+nc-1)
                
            else:
                self.mu[i] =  np.random.multivariate_normal(np.zeros(self.dim), np.diag(np.repeat(np.sqrt(self.nu), self.dim)))
                self.sigma[i] = np.diag(1/gamma(np.repeat(self.a, self.dim), np.repeat(self.b, self.dim)))
                
            
        total_loglikelihood = 0
        
        for i in range(self.n):
            emis = 0
            trans = 0
            for t in range(self.T):
                
                emis+=self._logphi(self.data[i, t], self.mu[self.state[i][t]], self.sigma[self.state[i][t]])
                if t>0:
                    trans+=np.log(self.PI[self.state[i][t-1], self.state[i][t]])
            total_loglikelihood = emis+trans
            #print("Total log likelihood of all sequences:", total_loglikelihood)
            
        
                
                
                
        
    def _gem(self, gma):
        prev = 1
        while True:
            beta_k = beta(1, gma)*prev
            prev -= beta_k
            yield beta_k
            

if __name__=="__main__":
    mean = np.zeros(16)
    cov = np.diag(np.repeat(np.sqrt(1), 16))
    data = []
    for i in range(20):
        mean= mean+1
        cov = cov+np.sqrt(i)
        temp = np.random.multivariate_normal(mean, cov, (32))
        data.append(temp)
    data = np.array(data)
    data = data.reshape(data.shape[1], data.shape[0], data.shape[2])
    print(data.shape)
        
    a = np.random.rand(5, 32, 16)
    sticky_hdp_hmm = StickyHDPHMM(a)
    N = sticky_hdp_hmm.N

    
    for i in range(100):
        sticky_hdp_hmm.sampler()
        print(sticky_hdp_hmm.state)
    
     
