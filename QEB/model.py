import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Union

random.seed(0)
np.random.seed(0)


class QEB:
    def create_data(
            self,
            mu_0: int = 2,
            sigma_0: int = 0.5,
            numbers_of_each_type: Dict[str, int] = {"A": 3, "B": 2},
            adjustment: Dict[str, List[int]] = {"A": [1] * 3, "B": [1.2] * 2},
            grid_points: np.ndarray = np.repeat(np.array([1, 2, 3]), 20),
            variances_of_error: Dict[str, List[int]] = {"A": [1] * 3, "B": [1.5] * 2},
    ) -> Dict[str, np.ndarray]:
        """
        Create simulated data with given hyper parameters

        Params
        ----------
        mu_0: int
            The `mu` of normal distribution in the first layer of bayesian
        sigma_0: int =0.5
            The `sigma` of normal distribution in the first layer of bayesian
        numbers_of_each_type: Dict[str, List[int]]
            The number of sample size in each type of amino acid
        adjustment: Dict[str, List[int]]
            The adjustment to fit gaussian in each type of amino acid
        grid_points: np.ndarray
            The distances of grid points to the center of amino acid
        variances_of_error: Dict[str, List[int]]
            The variances of errors which is also a gaussian distribution

        Return
        ----------
        self.data_log: Dict[str, np.ndarray]
            The transformed data by log

        """

        self.mu_0 = mu_0
        self.sigma_0 = sigma_0
        self.numbers_of_each_type = numbers_of_each_type
        self.adjustment = adjustment
        self.grid_points = grid_points
        self.variances_of_error = variances_of_error
        self.X_k = - 1 / 2 * grid_points**2
        self.types_of_proteins = len(numbers_of_each_type)

        self.beta = {
            key: np.abs(
                np.random.normal(
                    self.mu_0,
                    self.sigma_0)) for key in numbers_of_each_type.keys()}
        self.data_log = {}
        for type_j, sample_size in numbers_of_each_type.items():
            beta = self.beta[type_j]
            epsilon = np.array([np.random.normal(0, variance, grid_points.shape)
                               for variance in variances_of_error[type_j]])
            A_ij = adjustment[type_j]
            A_ij_tilde = -3 / 2 * np.log(2 * np.pi * 1 / beta) + np.log(A_ij)
            data_list = []
            for i in range(sample_size):
                data_list.append(A_ij_tilde[i] + beta * self.X_k + epsilon[i])
            self.data_log[type_j] = np.array(data_list)
        return self.data_log

    def caculate_constants(self) -> Tuple[Dict[str, np.ndarray]]:
        """
        Estimate `A_ij_tilde` by simple linear regression solved by OLS.
        Estimate `epsilon_ij` by MSE.
        Caculate the constants of `B_j`, `C_j`, `D_j` `sigma_j_bar` before caculate log likelihood function.

        Params
        ----------

        Return
        ----------
        Tuple[Dict[str, np.ndarray]]
            The constant values that would be used to caculate the bayes estimator.
        """
        self.estimated_A_ij_tilde = {key: np.array(
            []) for key in self.numbers_of_each_type.keys()}
        self.estimated_variances_of_error = {key: np.array(
            []) for key in self.numbers_of_each_type.keys()}
        self.B_j = {key: np.array([])
                    for key in self.numbers_of_each_type.keys()}
        self.C_j = {key: np.array([])
                    for key in self.numbers_of_each_type.keys()}
        self.D_j = {key: np.array([])
                    for key in self.numbers_of_each_type.keys()}
        self.sigma_j_bar = {
            key: np.array(
                []) for key in self.numbers_of_each_type.keys()}
        X_k_distance = self.X_k.mean()
        for type_j, sample_size in self.numbers_of_each_type.items():
            estimated_A_ij_tilde_list = []
            estimated_variance = []
            B_j_list = []
            C_j_list = []
            D_j_list = []
            sigma_j_bar = 1
            data_log_arr = self.data_log[type_j]
            for i in range(sample_size):
                data_log = data_log_arr[i]
                cov = np.cov(self.X_k, data_log)[0, 1]
                var = np.var(self.X_k)
                beta_ols = cov / var
                A_ij = data_log.mean() - beta_ols * X_k_distance
                variance = sum(
                    (data_log - (A_ij + self.X_k * beta_ols))**2) / (data_log.shape[0] - 1)
                estimated_A_ij_tilde_list.append(A_ij)
                estimated_variance.append(variance)
                # Sum `k`
                B_j_list.append(sum((data_log - A_ij)**2 / (2 * variance)))
                C_j_list.append(
                    sum((data_log - A_ij) * self.grid_points**2 / (2 * variance)))
                D_j_list.append(sum(self.grid_points**4 / (8 * variance)))
                sigma_j_bar *= variance
            self.estimated_A_ij_tilde[type_j] = np.array(
                estimated_A_ij_tilde_list)
            self.estimated_variances_of_error[type_j] = np.array(
                estimated_variance)
            # Sum `i`
            self.B_j[type_j] = -sum(B_j_list)
            self.C_j[type_j] = -sum(C_j_list)
            self.D_j[type_j] = -sum(D_j_list)
            self.sigma_j_bar[type_j] = sigma_j_bar

        return self.estimated_A_ij_tilde, self.B_j, self.C_j, self.D_j, self.sigma_j_bar

    def negative_log_likelihood(
            self, params: Tuple[Union[int, float]]) -> float:
        """
        The negative log likelihood function of `mu_0` and `sigma_0`.

        Params
        ----------
        params: Tuple[Union[int, float]]
            The parameters of negative log-likelihood are `mu` and `sigma`

        Return
        ----------
        float
            The value of negative log-likelihood given params
        """
        mu_0, sigma_0 = params
        B_j = np.array([value for value in self.B_j.values()])
        C_j = np.array([value for value in self.C_j.values()])
        D_j = np.array([value for value in self.D_j.values()])
        sigma_j_bar = np.array([value for value in self.sigma_j_bar.values()])
        E_j = 1 - 2 * sigma_0 * D_j
        return -sum(-1 / 2 * np.log(sigma_j_bar * E_j) - (sigma_0 * (4 * B_j * \
                    D_j - C_j**2) - 2 * mu_0 * C_j - 2 * B_j - 2 * mu_0**2 * D_j) / (2 * E_j))

    def estimate_hyperparameters_by_mle(self) -> Tuple[Union[int, float]]:
        """
        Estimate the hyperparameters, `mu_0` and `sigma_0`, of Gaussian distribution by mle.

        Params
        ----------

        Return
        ----------
        Tuple[Union[int, float]]
            The estimators of `mu_0` and `sigma_0`
        """
        self.res = minimize(self.negative_log_likelihood,
                            (2, 0.5), method='Nelder-Mead')
        self.mu_0_hat, self.sigma_0_hat = self.res.x
        return self.mu_0_hat, self.sigma_0_hat

    def estimate_beta_1j_by_Bayse(self) -> Dict[str, np.ndarray]:
        """
        Estimate `beta_1j` by bayes' estimator.

        Params
        ----------

        Return
        ----------
        Dict[str, np.ndarray]
            The bayes estimators of `beta_1j`
        """
        C_j = np.array([value for value in self.C_j.values()])
        D_j = np.array([value for value in self.D_j.values()])
        E_j = 1 - 2 * self.sigma_0_hat * D_j
        beta_1j_hat = (self.mu_0_hat + self.sigma_0_hat * C_j) / E_j
        self.beta_1j_hat = {
            key: value for key, value in zip(
                self.C_j.keys(), beta_1j_hat)}
        return self.beta_1j_hat
