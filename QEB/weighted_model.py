import numpy as np
import pandas as pd
import random
import logging
from scipy.optimize import minimize, fsolve
from scipy.special import digamma
from scipy.optimize import fsolve
from typing import List, Dict, Tuple, Optional
from .utils import *
from .calculation import *
from copy import copy

random.seed(10)
np.random.seed(10)
logging.getLogger().setLevel(logging.INFO)

class WQEB:
    def create_data(self, 
                    group_num = 5,
                    group_name = ["A", "B", "C", "D", "E"], 
                    in_group_num = 6,
                    contained_ratio_data = 0.1,
                    contained_ratio_beta = 0.1,
                    ):
        # hyperparameter 
        contained_group_num = int(in_group_num * contained_ratio_beta) 
        self.mus = {group_name[idx]: val for idx, val in enumerate(np.array([[-8, 8]] * group_num))}
        self.vs = {group_name[idx]: val for idx, val in enumerate([400] * group_num)}
        self.ss = {group_name[idx]: val for idx, val in enumerate([0.1] * group_num)}

        # grid points
        distances_to_center = np.repeat(np.arange(0.01, 0.51, 0.01), 10)
        grid_num = len(distances_to_center)
        X_tilde = [[np.concatenate((np.ones((grid_num, 1)), (-1/ 2 * distances_to_center ** 2).reshape(-1, 1)), axis=1).tolist()] * in_group_num]
        self.distances_to_center = {group_name[idx]: val for idx, val in enumerate(np.array([[distances_to_center]* in_group_num] * group_num))}
        self.Xs_tilde = {group_name[idx]: val for idx, val in enumerate(np.array(X_tilde * group_num))}
        self.lambdas = {group_name[idx]: val for idx, val in enumerate([np.ones(in_group_num)] * group_num)}
        self.weight_matrixs = {group_name[idx]: val for idx, val in enumerate(np.array([[np.eye(grid_num)] * in_group_num] * group_num))}

        # prior
        sigmas = []
        for v, s in zip(self.vs.values(), self.ss.values()):
            sigmas.append(invgamma.rvs(a=v / 2, scale=v * s / 2, size=in_group_num))
        self.sigmas = {group_name[idx]: val for idx, val in enumerate(np.array(sigmas))}

        betas = []
        for mu, sigma_all, lambda_all in \
            zip(self.mus.values(), self.sigmas.values(), self.lambdas.values()):
            beta = []
            counts = contained_group_num
            for sigma, lambda_ in zip(sigma_all, lambda_all):
                if counts > 0:
                    beta.append(multivariate_normal.rvs([-5, 5], sigma * lambda_ ** (-1) * np.eye(2)))
                    counts -= 1
                else:
                    beta.append(multivariate_normal.rvs(mu, sigma * lambda_ ** (-1) * np.eye(2)))
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
                if contained_ratio_data > 0:
                    index = np.random.choice(np.arange(0, grid_num), int(grid_num * contained_ratio_data), replace=False)
                    # contaminated_num = int(grid_num * contained_ratio_data)
                    # radius_num = int((0.51 - 0.01) / 0.01) + 1
                    # print(radius_num)
                    # print(contaminated_num)
                    # one_time = np.random.choice(np.arange(0, 10), contaminated_num // radius_num, replace=False)
                    # index = [num + 10 * i for num in one_time for i in range(radius_num)]
                    # print(len(index))
                    # index = np.random.choice(index, contaminated_num)
                    # print(len(index))
                    data[index] = multivariate_normal.rvs(X_tilde[index, :] @ np.array([-1, 1]), sigma * weight_matrix[index, index])
                else:
                    index = [] 
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
        residue_names = np.array(df_processed["residue_name"])
        atom_points = np.column_stack((df_processed.x_coord, df_processed.y_coord, df_processed.z_coord))
        self.grid_points, self.distances_to_center, self.Xs_tilde = generate_grid_points(
            atom_points, residue_names, start_rad, max_rad, gap, max_points, base_num_points, max_iter)
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
            mus_initial.append(np.median(mu_initial, axis=0))
        self.mus_initial = {name: val for name, val in zip(self.data_log, mus_initial)}

        self.sigmas_initial = caluculate_sigmas(self.Xs_tilde, self.data_log, self.mus_initial, initial=True)

        self.weight_matrixs = {}
        for name in self.Xs_tilde:
            weight_matrix_new = []
            for X_tilde in self.Xs_tilde[name]:
                weight_matrix_new.append(np.eye(len(X_tilde)))
            self.weight_matrixs[name] = np.array(weight_matrix_new)
        self.lambdas = {name: np.ones(len(self.Xs_tilde[name])) for name in self.Xs_tilde}
        
        return self.mus_initial, self.sigmas_initial, self.weight_matrixs, self.lambdas

    def algorithm_iter(self, max_iter = 3, alpha = 0.1, gamma = 0.1, tol = 1e-4, patience = 3, verbose = True):
        # self.betas_WEB = caculate_mus_tilde(
        #     self.Xs_tilde, self.data_log, self.weight_matrixs, self.lambdas, self.mus_initial)
        self.mus_mle = self.mus_initial
        self.sigmas_median  = self.sigmas_initial
        self.densities_data = density_mean([self.data], self.distances_to_center)[0]
        best_similarity = -np.inf
        least_difference = np.inf
        cur_betas = self.mus_initial
        self.beta_histories = []
        self.beta_differences_histories = []
        iter_num = 0
        for i in tqdm(range(max_iter), disable = not verbose):
            self.betas_WEB = caculate_mus_tilde(self.Xs_tilde, self.data_log, self.weight_matrixs, self.lambdas, self.mus_mle)
            self.mus_mle = caculate_mus_mle(self.Xs_tilde, self.data_log, self.sigmas_median, self.weight_matrixs, self.lambdas)
            self.sigmas_median = caluculate_sigmas(self.Xs_tilde, self.data_log, self.betas_WEB)
            self.weight_matrixs, self.lambdas = caculate_weights_and_lamdas(
                self.Xs_tilde, self.data_log, self.betas_WEB, self.sigmas_median, alpha, gamma, self.mus_mle)
            # self.weight_matrixs, self.lambdas = caculate_weights_and_lamdas(
            #     self.Xs_tilde, self.data_log, self.betas_WEB, self.sigmas_median , self.mus_mle, alpha, gamma)
            # self.mus_mle = caculate_mus_mle(self.Xs_tilde, self.data_log, self.sigmas_median, self.weight_matrixs, self.lambdas)
            # self.sigmas_median = caluculate_sigmas(self.Xs_tilde, self.data_log, self.mus_mle, initial=True)
            # self.betas_WEB = caculate_mus_tilde(self.Xs_tilde, self.data_log, self.weight_matrixs, self.lambdas, self.mus_mle)
            
            self.points_betas_WEB = caculate_points(self.Xs_tilde, self.betas_WEB)
            self.densities_betas_WEB = density_mean([self.points_betas_WEB], self.distances_to_center)[0]
            self.similarities = caculate_similarity(self.densities_data, self.densities_betas_WEB)
            # similarity_all = [np.mean(similarity) for similarity in self.similarities.values()]
            similarity_all = []
            for similarity in self.similarities.values():
                similarity_all.extend(similarity)

            beta_differences = []
            for new_beta, cur_beta in zip(cur_betas.values(), self.betas_WEB.values()):
                beta_differences.extend((new_beta - cur_beta) ** 2)
            if verbose:
                logging.info(f"Iteration {i} finished. with difference: {np.mean(beta_differences)}")
                # print(len(np.where(np.array(similarity_all) < 0.95)[0]))
                # sum_weights = {name: np.sum([np.diag(weight_matrix) for weight_matrix in self.weight_matrixs[name]], axis=1) for name in self.weight_matrixs}
                # logging.info(f"Iteration {i} finished. with weights: {sum_weights}, lambdas: {self.lambdas}")
            
            # if np.mean(similarity_all) > best_similarity:
            #     best_similarity = np.mean(similarity_all)
            iter_num += 1
            if np.mean(beta_differences) > tol and iter_num < (patience + 1):
                if np.mean(beta_differences) < least_difference:
                    least_difference = np.mean(beta_differences)
                    iter_num = 0
                cur_betas = self.betas_WEB
                self.beta_histories.append(self.betas_WEB)
                self.beta_differences_histories.append(np.mean(beta_differences))
            else:
                break
        return self.betas_WEB, np.mean(beta_differences)

    def weighted_linear_regression(self, iter_num = 3, alpha = 0.1, gamma = 0.1, verbose = True):
        self.sigmas_median  = self.sigmas_initial
        self.densities_data = density_mean([self.data], self.distances_to_center)[0]
        best_similarity = -np.inf
        for i in tqdm(range(iter_num), disable=not verbose):
            self.betas_weighted = caculate_weighted_beta(self.Xs_tilde, self.data_log, self.weight_matrixs)
            self.sigmas_median = caluculate_sigmas(self.Xs_tilde, self.data_log, self.betas_weighted)
            self.weight_matrixs, Q = caculate_weights_and_lamdas(
                self.Xs_tilde, self.data_log, self.betas_weighted, self.sigmas_median, alpha, gamma)
        
            self.points_betas_weighted = caculate_points(self.Xs_tilde, self.betas_weighted)
            self.densities_betas_weighted = density_mean([self.points_betas_weighted], self.distances_to_center)[0]
            self.similarities = caculate_similarity(self.densities_data, self.densities_betas_weighted)
            similarity_all = []
            for similarity in self.similarities.values():
                similarity_all.extend(similarity)
            if verbose:
                logging.info(f"Iteration {i} finished. with similarity: {np.min(similarity_all)}")
                # print(len(np.where(np.array(similarity_all) < 0.95)[0]))
                # sum_weights = {name: np.sum([np.diag(weight_matrix) for weight_matrix in self.weight_matrixs[name]], axis=1) for name in self.weight_matrixs}
                # logging.info(f"Iteration {i} finished. with weights: {sum_weights}, lambdas: {self.lambdas}")
            
            # if np.min(similarity_all) > best_similarity:
            #     best_similarity = np.min(similarity_all)
            # else:
            #     break

        return self.betas_weighted
    
    
    def plot_data(self, max_radius, gap, save=False):
        self.betas_em_mean = {name: np.mean(betas, axis=0) for name, betas in self.betas_WEB.items() if len(betas) > 0}
        # self.betas_em_weighted_mean = {betas_item[0]: np.sum(np.repeat(lambda_, 2).reshape(-1, 2) * betas_item[1], axis=0) / np.sum(lambda_) \
        #                                for betas_item, lambda_ in zip(self.betas_WEB.items(), self.lambdas.values()) if len(betas_item[1]) > 0}
        
        self.densities_mle = caculate_density(self.distances_to_center, self.mus_mle)
        self.densities_em = caculate_density(self.distances_to_center, self.betas_em_mean)
        # self.densities_em_weighted = caculate_density(self.distances_to_center, self.betas_em_weighted_mean)

        plot_density(self.densities_data, 
                     [self.densities_mle, self.densities_em], 
                     max_radius, 
                     gap, 
                     ["MLE", "WEB mean"], 
                     ["blue", "red"], 
                     save=save)
        