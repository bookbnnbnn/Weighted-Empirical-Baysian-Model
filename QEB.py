import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import multivariate_normal
from typing import List, Dict, Tuple, Optional

random.seed(0)
np.random.seed(0)

class QEB:
    def __init__(
        self, 
        mu_0: Optional[int]=None, 
        sigma_0: Optional[int]=None, 
        types_of_proteins: Optional[int]=None, 
        numbers_of_each_type: Optional[List]=None, 
        adjustment_list: Optional[List[List]]=None, 
        grid_points: Optional[np.ndarray]=None,
        variances_of_error: Optional[List]=None, 
        beta_list: Optional[List]=None, 
        epsilon_list: Optional[List[np.ndarray]]=None,
        data_log_list: Optional[List[np.ndarray]]=None,
    ) -> None:
        self.mu_0 = 0 if mu_0 is None else mu_0
        self.sigma_0 = 1 if sigma_0 is None else sigma_0
        self.types_of_proteins = 1 if types_of_proteins is None else types_of_proteins
        self.numbers_of_each_type = [] if numbers_of_each_type is None else numbers_of_each_type
        self.adjustment_list = [] if adjustment_list is None else adjustment_list
        self.grid_points = np.array([]) if grid_points is None else grid_points
        self.variances_of_error = [] if variances_of_error is None else variances_of_error
        self.beta_list = [] if beta_list is None else beta_list
        self.epsilon_list = [] if epsilon_list is None else epsilon_list
        self.data_log_list = [] if data_log_list is None else data_log_list
        
        self.X_k = np.array([]) if grid_points is None else -1/2*grid_points**2

    def create_data(self) -> np.ndarray:
        """Create simulated data with given hyper parameters"""

        for j in range(self.types_of_proteins):
            data_log_list = []
            epsilon_list = []
            beta = np.abs(np.random.normal(self.mu_0, self.sigma_0))
            self.beta_list.append(beta)
            self.numbers_of_types = self.numbers_of_each_type[j]
            for i in range(self.numbers_of_types):
                A_ij = self.adjustment_list[j][i]
                sigma_ij = self.variances_of_error[j][i]
                epsilon = np.random.normal(0, sigma_ij, self.grid_points.shape)
                A_ij_tilde = -3/2 * np.log(2*np.pi * 1/beta) + np.log(A_ij)
                data_log = A_ij_tilde + beta*self.X_k + epsilon
                data_log_list.append(data_log)
                epsilon_list.append(epsilon)
            self.epsilon_list.append(epsilon_list)
            self.data_log_list.append(np.array(data_log_list))

        return self.data_log_list
    
    def estimate_A_ij_tilde(self) -> np.ndarray:
        """Estimate `A_ij_tilde` by simple linear regression solved by OLS"""
        beta_ols = [
            [np.cov(self.X_k, self.data_log_list[j][i])[0, 1] / np.var(self.X_k) for i in range(self.numbers_of_each_type[j])]
            for j in range(self.types_of_proteins)
        ]
        self.estimated_A_ij_tilde = [
            [self.data_log_list[j][i].mean() - beta_ols[j][i]*self.X_k.mean() for i in range(self.numbers_of_each_type[j])]
            for j in range(self.types_of_proteins) 
        ]
        return self.estimated_A_ij_tilde
    
    def caculate_constants(self) -> Tuple[np.ndarray]:
        """Caculate the constants of `B_j`, `C_j`, `D_j` before caculate log likelihood function."""
        self.B_j = np.array([])
        self.C_j = np.array([])
        self.D_j = np.array([])
        self.sigma_j_bar = np.array([])
        for j in range(self.types_of_proteins): 
            B_j_list = []
            C_j_list = []
            D_j_list = []
            sigma_j_bar = 1
            for i in range(self.numbers_of_each_type[j]):
                # Sum `k`
                B_j_list.append(sum((self.data_log_list[j][i] - self.estimated_A_ij_tilde[j][i])**2 / (2*qeb.variances_of_error[j][i])))
                C_j_list.append(sum((self.data_log_list[j][i] - self.estimated_A_ij_tilde[j][i]) * self.grid_points**2  / (2*qeb.variances_of_error[j][i])))
                D_j_list.append(sum(self.grid_points**4  / (8*qeb.variances_of_error[j][i])))
                sigma_j_bar *= self.variances_of_error[j][i]
            # Sum `i`
            self.B_j = np.append(self.B_j, -sum(B_j_list))
            self.C_j = np.append(self.C_j, -sum(C_j_list))
            self.D_j = np.append(self.D_j, -sum(D_j_list))
            self.sigma_j_bar = np.append(self.sigma_j_bar, sigma_j_bar)
        return self.B_j, self.C_j, self.D_j, self.sigma_j_bar
        
    def negative_log_likelihood(self, params: Tuple) -> float:
        """The negative log likelihood function of `mu_0` and `sigma_0`."""
        mu_0, sigma_0 = params
        E_j = 1 - 2*sigma_0*self.D_j
        return -sum(-1/2* np.log(self.sigma_j_bar * E_j) - (sigma_0 * (4*self.B_j*self.D_j - self.C_j**2) - 2*mu_0*self.C_j - 2*self.B_j - 2*mu_0**2*self.D_j) / (2*E_j))
     
    def estimate_hyperparameters_by_mle(self) -> Tuple: 
        """Estimate the hyperparameters, `mu_0` and `sigma_0`, of Gaussian distribution by mle."""
        self.res = minimize(self.negative_log_likelihood, (-1, 0.1), method='Nelder-Mead')
        self.mu_0_hat, self.sigma_0_hat = self.res.x
        return self.mu_0_hat, self.sigma_0_hat
    
    def estimate_beta_1j_by_Bayse(self) -> np.ndarray:
        """Estimate `beta_1j` by bayes' estimator."""
        self.E_j = 1 - 2*self.sigma_0_hat*self.D_j
        self.beta_1j_hat = (self.mu_0_hat + self.sigma_0_hat*self.C_j) / self.E_j
        return self.beta_1j_hat