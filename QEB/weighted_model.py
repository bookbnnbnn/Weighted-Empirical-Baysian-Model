import numpy as np
import pandas as pd
import random
import logging
from scipy.optimize import minimize, fsolve
from scipy.special import digamma
from scipy.optimize import fsolve
from typing import List, Dict, Tuple, Optional
from .utils import *
from .caculation import *
from copy import copy

random.seed(0)
np.random.seed(0)
logging.getLogger().setLevel(logging.INFO)

class WQEB:
    def create_data(self, 
                    group_num = 5,
                    group_name = ["A", "B", "C", "D", "E"], 
                    in_group_num = 5,
                    contained_ratio = 0.01,
                    ):
        # hyperparameter 
        self.mus = {group_name[idx]: val for idx, val in enumerate(np.array([[-1.5, 5]] * group_num))}
        A = np.random.uniform(5, 10, (group_num, 2, 2))
        sigma_matrixs = []
        for A in np.random.rand(group_num, 2, 2):
            # sigma_matrixs.append(A.T @ A)
            sigma_matrixs.append(np.eye(2))
        self.sigma_matrixs = {group_name[idx]: val for idx, val in enumerate(sigma_matrixs)}

        self.vs = {group_name[idx]: val for idx, val in enumerate([400] * group_num)}
        self.ss = {group_name[idx]: val for idx, val in enumerate([0.0001] * group_num)}

        # grid points
        distances_to_center = np.repeat(np.arange(0.01, 0.5, 0.01), 10)
        grid_num = len(distances_to_center)
        X_tilde = [[np.concatenate((np.ones((grid_num, 1)), (-1/ 2 * distances_to_center ** 2).reshape(-1, 1)), axis=1).tolist()] * in_group_num]
        self.distances_to_center = {group_name[idx]: val for idx, val in enumerate(np.array([[distances_to_center]* in_group_num] * group_num))}
        self.Xs_tilde = {group_name[idx]: val for idx, val in enumerate(np.array(X_tilde * group_num))}
        self.lambda_matrixs = {group_name[idx]: val for idx, val in enumerate([np.eye(2)] * group_num)}
        self.weight_matrixs = {group_name[idx]: val for idx, val in enumerate(np.array([[np.eye(grid_num)] * in_group_num] * group_num))}

        # prior
        sigmas = []
        for v, s in zip(self.vs.values(), self.ss.values()):
            sigmas.append(invgamma.rvs(a=v / 2, scale=v * s / 2, size=in_group_num))
        self.sigmas = {group_name[idx]: val for idx, val in enumerate(np.array(sigmas))}

        betas = []
        for mu, sigma_all, sigma_matrix, lambda_matrix in \
            zip(self.mus.values(), self.sigmas.values(), self.sigma_matrixs.values(), self.lambda_matrixs.values()):
            beta = []
            for sigma in sigma_all:
                beta.append(multivariate_normal.rvs(mu, sigma * np.linalg.inv(lambda_matrix) @ np.linalg.inv(sigma_matrix)))
            betas.append(np.array(beta))
        self.betas = {group_name[idx]: val for idx, val in enumerate(betas)}

        # data
        ys_tilde = []
        ys_tilde_clean = []
        for beta_all, sigma_all, weight_matrix_all, X_tilde_all in \
            zip(self.betas.values(), self.sigmas.values(), self.weight_matrixs.values(), self.Xs_tilde.values()):
            y_tilde = []
            y_tilde_clean = []
            for beta, sigma, weight_matrix, X_tilde in zip(beta_all, sigma_all, weight_matrix_all, X_tilde_all):
                data = multivariate_normal.rvs(X_tilde @ beta, sigma * weight_matrix)
                index = np.random.choice(np.arange(0, grid_num), int(grid_num * contained_ratio), replace=False)
                data[index] += np.random.uniform(0.5, 1, size=len(index))
                y_tilde.append(data)
                clean_data = copy(data)
                clean_data[index] = np.nan
                y_tilde_clean.append(np.array(clean_data))
            ys_tilde.append(np.array(y_tilde))
            ys_tilde_clean.append(np.array(y_tilde_clean))
        self.data_log = {group_name[idx]: val for idx, val in enumerate(ys_tilde)}
        self.data = {group_name[idx]: np.exp(val) for idx, val in enumerate(ys_tilde_clean)}
        return self.data_log

    def read_data(
            self,
            root_map: str,
            root_pdb: str,
            atomic: Optional[str] = None,
            start_rad: float = 0.01,
            max_rad: float = 1,
            gap: float = 0.01,
            max_points: int = 8,
            base_num_points: int = 4,
            max_iter: int = 30
    ) -> Dict[str, np.ndarray]:
        """
        Reads map and pdb files, generates grid points, interpolates data, and returns the log values of the
        interpolated data for each atom type.

        Params
        ----------
        root_map: str 
            Path to the map file.
        root_pdb: str 
            Path to the pdb file.
        atomic: Optional[str]
            Atom type to filter by. Defaults to None.
        start_rad: float (default=0.01)
            Minimum radius for generating grid points. 
        max_rad: float (default=0.8)
            Maximum radius for generating grid points. 
        gap: float (default=0.01)
            Gap between radii for generating grid points. 
        max_points: int (default=8)
            Maximum number of points to generate at each radius. 
        base_num_points: int (default=4)
            Number of points to generate at the minimum radius. 
        max_iter: int (default=30)
            Maximum number of iterations for generating grid points.

        Return
        ----------
        Dict[str, np.ndarray]: 
            the dictionary containing the log values of the interpolated data for each atom type.
        """
        data, grid_size, origin = read_map(root_map)
        mean, std = np.mean(data), np.std(data)
        self.A_B = mean + 10 * std, mean - std
        df_processed = read_pdb(root_pdb, atomic=atomic)
        self.grid_points, self.distances_to_center, self.Xs_tilde = generate_grid_points(
            df_processed, start_rad, max_rad, gap, max_points, base_num_points, max_iter)
        self.interp_func = interpolator(data, grid_size, origin)
        self.data = {key: self.interp_func(grid_points) for key, grid_points in self.grid_points.items()}
        self.data_log = {key: np.log(value + 1e-35) for key, value in self.data.items()}
        self.numbers_of_each_type = {atom: len(self.data_log[atom]) for atom in self.data_log.keys()}
        return self.data_log

    def paramters_initial(self):
        mus_initial = []
        for X_tilde_all, y_tilde_all in zip(self.Xs_tilde.values(), self.data_log.values()):
            mu_initial = []
            for X_tilde, y_tilde in zip(X_tilde_all, y_tilde_all):
                mu_initial.append(np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y_tilde)
            mus_initial.append(np.mean(mu_initial, axis=0))
        self.mus_initial = {name: val for name, val in zip(self.data_log, mus_initial)}

        self.sigma_matrixs_initial = {name: val for name, val in zip(self.data_log, [np.eye(2)] * len(self.data_log))}

        self.vs_initial = {name: 2 for name in self.data_log}

        ss_initial = []
        for X_tilde_all, y_tilde_all, mu, v in \
            zip(self.Xs_tilde.values(), self.data_log.values(), self.mus_initial.values(), self.vs_initial.values()):
            s_initial = []
            for X_tilde, y_tilde in zip(X_tilde_all, y_tilde_all):
                s_initial.append((y_tilde - X_tilde @ mu).T @ (y_tilde - X_tilde @ mu) / v)
            ss_initial.append(np.mean(s_initial, axis=0))
        self.ss_initial = {name: val for name, val in zip(self.data_log, ss_initial)}

        self.weight_matrixs = {}
        for name in self.Xs_tilde:
            weight_matrix_new = []
            for X_tilde in self.Xs_tilde[name]:
                weight_matrix_new.append(np.eye(len(X_tilde)))
            self.weight_matrixs[name] = np.array(weight_matrix_new)
        self.lambda_matrixs = {name: np.eye(2) for name in self.Xs_tilde}
        
        return self.mus_initial, self.sigma_matrixs_initial, self.vs_initial, self.ss_initial, self.weight_matrixs, self.lambda_matrixs

    def algorithm_iter(self, iter_num = 5, alpha = 0.1):
        mus_tilde, mus_mle, sigma_matrixs_tilde, as_tilde, bs_tilde = caculate_mu_mle(
            self.Xs_tilde, self.data_log, self.weight_matrixs, self.lambda_matrixs, \
                self.sigma_matrixs_initial, self.mus_initial, self.vs_initial, self.ss_initial
                )
        self.betas_em = {}
        self.sigmas_em = {}
        self.mus_mle = {}
        self.vs_mle = {}
        self.ss_mle = {}
        for name in tqdm(list(self.Xs_tilde)):
            vs = []
            ss = []
            v = self.vs_initial[name]
            s = self.ss_initial[name]
            sigma_matrix = np.eye(2)
            for i in range(iter_num):
                objective_func = lambda params: estimator_negative_log_likelihood(        
                params, self.Xs_tilde, self.data_log, self.lambda_matrixs, self.weight_matrixs, mus_mle, name
                )

                bs_tilde_all = bs_tilde
                grid_num = len(self.Xs_tilde[name][0])
                def gradient(params):
                    v, s = params[0], params[1]
                    return sum(v / 2 * (1 / s - (grid_num + v) / (v * s + 2 * (bs_tilde_all[name] - v * s / 2)))), \
                        sum(1 / 2 * np.log(v * s / 2) + 1 / 2 - digamma(v / 2) + digamma((v + grid_num) / 2) - \
                            1 / 2 * np.log(bs_tilde_all[name]) + (grid_num + v) / 4 * s / bs_tilde_all[name])
                
                # result = fsolve(gradient, [500, 1])


                # res = minimize(objective_func, (1, 1, 1, 1), method='L-BFGS-B', bounds=((0.1, None), (0.1, None), (0, None), (0, None)))
                res = minimize(objective_func, (1, 1), method='L-BFGS-B', bounds=((0, None), (0, None)), jac=gradient)

                # sigma_matrix = np.diag((res.x[:2]))
                v = res.x[0]
                s = res.x[1]
                logging.info("[Minimize res] " + "v: " + str(v) + " s: " + str(s))
                vs.append(v)
                ss.append(s)
                
                mu_tilde_all, mu_mle, sigma_matrixs_tilde_all, as_tilde_all, bs_tilde_all = caculate_mu_mle_single(
                    self.Xs_tilde[name], self.data_log[name], self.weight_matrixs[name], self.lambda_matrixs[name], \
                        sigma_matrix, self.mus_initial[name], v, s, name
                    )
                betas_em, sigmas_em = empirical_bayes_single(mu_tilde_all, as_tilde_all, bs_tilde_all)
                sigma_matrix = (betas_em - betas_em.mean()).T @ (betas_em - betas_em.mean())
                print(np.all(np.linalg.eigvals(sigma_matrix) > 0))
                print(sigma_matrix)
                # loss.append(sum((mu_mle[name] - mus[name]) ** 2))
                weight(self.weight_matrixs[name], self.Xs_tilde[name], self.data_log[name], betas_em, sigmas_em, alpha)
                
                # print(mu_mle)
                # print(sigma_matrix)
                # print(v)
                # print(s)
                # print("-"*10)
            self.betas_em[name] = betas_em
            self.sigmas_em[name] = sigmas_em
            self.mus_mle.update(mu_mle)
            self.vs_mle[name] = vs
            self.ss_mle[name] = ss

        return self.betas_em, self.sigmas_em
    
    def caculate_constants(self) -> Tuple[Dict[str, np.ndarray]]:
        """
        Estimate `beta_j` by OLS for the first iteration and replaced by bayes estimator for the rest of iteration.
        Estimate `A_ij_tilde` by simple linear regression solved by OLS.
        Caculate the constants of `B_j`, `C_j`, and `D_j`before caculate log likelihood function.

        Params
        ----------
        None
        
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
        for type_j, sample_size in self.numbers_of_each_type.items():
            estimated_A_ij_tilde_list = []
            estimated_variance = []
            B_j_list = []
            C_j_list = []
            D_j_list = []
            beta_estimators_list = []
            data_log_arr = self.data_log[type_j]
            center_distances_arr = self.distances_to_center[type_j]
            if None in self.beta_estimators.values():
                flatten_data_log_arr = [
                    item for sublist in data_log_arr for item in sublist
                    ] if isinstance(data_log_arr[0], np.ndarray) else data_log_arr
                flatten_center_distances_list = [
                    item for sublist in center_distances_arr for item in sublist
                    ] if isinstance(center_distances_arr[0], np.ndarray) else center_distances_arr
                all_data_log = np.array(flatten_data_log_arr)
                all_X_k = - 1 / 2 * np.array(flatten_center_distances_list)**2
                cov = np.cov(all_X_k, all_data_log)[0, 1]
                var = np.var(all_X_k)
                beta_estimators = cov / var
                self.beta_estimators[type_j] = beta_estimators
            else:
                beta_estimators = self.beta_estimators[type_j]

            for i in range(sample_size):
                data_log = data_log_arr[i]
                center_distances = center_distances_arr[i]
                X_k = - 1 / 2 * center_distances**2
                A_ij = data_log.mean() - beta_estimators * X_k.mean()
                variance = sum((data_log - (A_ij + X_k * beta_estimators))
                               ** 2) / (data_log.shape[0] - 1)
                beta_estimators_list.append(beta_estimators)
                estimated_A_ij_tilde_list.append(A_ij)
                estimated_variance.append(variance)
                # Sum `k`
                B_j_list.append(sum((data_log - A_ij)**2 / (2 * variance)))
                C_j_list.append(
                    sum((data_log - A_ij) * center_distances**2 / (2 * variance)))
                D_j_list.append(sum(center_distances**4 / (8 * variance)))
            self.estimated_A_ij_tilde[type_j] = np.array(
                estimated_A_ij_tilde_list)
            self.estimated_variances_of_error[type_j] = np.array(
                estimated_variance)
            # Sum `i`
            self.B_j[type_j] = -sum(B_j_list)
            self.C_j[type_j] = -sum(C_j_list)
            self.D_j[type_j] = -sum(D_j_list)

        return self.beta_estimators, self.estimated_A_ij_tilde, self.B_j, self.C_j, self.D_j

    def negative_log_likelihood(
            self, params: Tuple[float]) -> float:
        """
        The negative log likelihood function of `mu_0` and `sigma_0`.

        Params
        ----------
        params: Tuple[float]
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
        E_j = 1 - 2 * sigma_0 * D_j
        mu_tilde = (mu_0 + sigma_0 * C_j) / E_j
        sigma_tilde = sigma_0/ (1 - 2 * sigma_0 * D_j)
        truncated = (1 - norm.cdf(-mu_tilde / sigma_tilde)) - (1 - norm.cdf(mu_0 / sigma_0))
        return -sum(-1 / 2 * (np.log(E_j)) - (sigma_0 * (4 * B_j * \
                    D_j - C_j**2) - 2 * mu_0 * C_j - 2 * B_j - 2 * mu_0**2 * D_j) / (2 * E_j) + truncated)

    def estimate_hyperparameters_by_mle(self) -> Tuple[float]:
        """
        Estimate the hyperparameters, `mu_0` and `sigma_0`, of Gaussian distribution by mle.

        Params
        ----------
        None

        Return
        ----------
        Tuple[float]
            The estimators of `mu_0` and `sigma_0`
        """
        self.res = minimize(self.negative_log_likelihood,
                            (2, 0.5), method='Nelder-Mead', bounds=((None, None), (1e-20, None)))
        self.mu_0_hat, self.sigma_0_hat = self.res.x
        return self.mu_0_hat, self.sigma_0_hat

    def estimate_beta_j_by_Bayse(
            self,
            iteration: int = 30, 
            tolerate: float = 0.000001
    ) -> Dict[str, np.ndarray]:
        """
        Estimate `beta_j` by bayes' estimator.

        Params
        ----------
        iteration: int
            The number of how many times we want to iterate
        tolerate: float
            The minimal error that we can tolerate

        Return
        ----------
        Dict[str, np.ndarray]
            The bayes estimators of `beta_j`
        """

        self.beta_estimators = {key: None for key in self.numbers_of_each_type.keys()}
        mse_list = []
        for _ in range(iteration):
            self.caculate_constants()
            self.estimate_hyperparameters_by_mle()
            beta_j_hat_last = np.array([value for value in self.beta_estimators.values()])
            C_j = np.array([value for value in self.C_j.values()])
            D_j = np.array([value for value in self.D_j.values()])
            E_j = 1 - 2 * self.sigma_0_hat * D_j
            mu_tilde = (self.mu_0_hat + self.sigma_0_hat * C_j) / E_j
            sigma_tilde = self.sigma_0_hat / (1 - 2 * self.sigma_0_hat * D_j)
            a = -mu_tilde / np.sqrt(sigma_tilde)
            beta_j_hat = mu_tilde + norm.pdf(a) / (1 - norm.cdf(a)) * np.sqrt(sigma_tilde)
            mse = sum((beta_j_hat_last - beta_j_hat)**2)
            self.beta_estimators = {name: beta for name, beta in zip(self.beta_estimators, beta_j_hat)}
            mse_list.append(mse)
            if mse < tolerate:
                break
        self.beta_j_hat = {key: value for key, value in zip(self.C_j.keys(), beta_j_hat)}
        return self.beta_j_hat, mse_list


    def plot_data(
            self, 
            amino_acid: Optional[str] = None, 
            indexes: Optional[List[int]] = None, 
            start_rad: float = 0.01,
            max_rad: float = 0.8,
            gap: float = 0.01,
            compared: bool = False,
            estimated: bool = True
        ) -> None:
        """
        Plot the density data.

        Params
        ----------
        amino_acid: Optional[str]
            The type of amino acid to plot data for. If None, plot data for all types.
        indexes: Optional[List[int]]
            The indexes of the data points to plot. If None, plot all points.
        start_rad: float
            The starting radius from the center. 
        max_rad: float
            The maximum radius from the center. 
        gap: float
            The gap between two radius.
        compared: bool
            Whether to compare the estimated Gaussian with the Gaussian in qscore.
        estimated: bool
            Whether to plot the estimated Gaussian.
            
        Return
        ----------
        None
        """
        # self.radius_density, self.estimated_radius_density, self.weighted_estimated_radius_density, self.qscore_radius_density = estimate_density(
        #     self.data, self.Xs_tilde, self.betas_em, self.beta_j_hat, self.estimated_A_ij_tilde)
        # plot_density(self.radius_density, self.estimated_radius_density, self.weighted_estimated_radius_density, 
        #              self.qscore_radius_density, amino_acid, indexes, start_rad, max_rad, gap, compared, estimated)
        
        self.radius_density, self.weighted_estimated_radius_density, self.qscore_radius_density = estimate_density_new(self.data, self.Xs_tilde, self.betas_em)
        plot_density(self.radius_density, self.weighted_estimated_radius_density, 
                     self.qscore_radius_density, amino_acid, indexes, start_rad, max_rad, gap, compared, estimated)