import numpy as np
import pandas as pd
import random
import logging
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
from .utils import *
from .caculation import *

random.seed(0)
np.random.seed(0)
logging.getLogger().setLevel(logging.INFO)

class WQEB:

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
        for name in self.distances_to_center:
            weight_matrix_new = []
            for distance_to_center in self.distances_to_center[name]:
                weight_matrix_new.append(np.eye(len(distance_to_center)))
            self.weight_matrixs[name] = np.array(weight_matrix_new)
        self.lambda_matrixs = {name: np.eye(2) for name in self.distances_to_center}
        
        return self.mus_initial, self.sigma_matrixs_initial, self.vs_initial, self.ss_initial, self.weight_matrixs, self.lambda_matrixs

    def algorithm_iter(self, iter_num = 5, alpha = 0.1):
        mus_tilde, mus_mle, sigma_matrixs_tilde, as_tilde, bs_tilde = caculate_mu_mle(
            self.Xs_tilde, self.data_log, self.weight_matrixs, self.lambda_matrixs, \
                self.sigma_matrixs_initial, self.mus_initial, self.vs_initial, self.ss_initial
                )
        self.betas_em = {}
        self.sigmas_em = {}
        for name in tqdm(list(self.distances_to_center)):
            for i in range(iter_num):
                objective_func = lambda params: estimator_negative_log_likelihood(        
                params, self.Xs_tilde, self.data_log, self.lambda_matrixs, self.weight_matrixs, mus_mle, name
                )

                logging.info("Minimize res")
                start = time.time()
                # res = minimize(objective_func, (1, 1, 1, 1), method='L-BFGS-B', bounds=((0.1, None), (0.1, None), (0, None), (0, None)))
                res = minimize(objective_func, (1, 1), method='L-BFGS-B', bounds=((0, None), (0, None)))
                end = time.time()
                logging.info("Spending Time: " + str(round(end - start, 2)))

                # sigma_matrix = np.diag((res.x[:2]))
                sigma_matrix = np.eye(2)
                v = res.x[0]
                s = res.x[1]
                
                logging.info("Caculate mu mle")
                start = time.time()
                mu_tilde_all, mu_mle, sigma_matrixs_tilde_all, as_tilde_all, bs_tilde_all = caculate_mu_mle_single(
                    self.Xs_tilde[name], self.data_log[name], self.weight_matrixs[name], self.lambda_matrixs[name], \
                        sigma_matrix, self.mus_initial[name], v, s, name
                    )
                end = time.time()
                logging.info("Spending Time: " + str(round(end - start, 2)))
                
                logging.info("Caculate Empirical Bayes")
                start = time.time()
                betas_em, sigmas_em = empirical_bayes_single(mu_tilde_all, as_tilde_all, bs_tilde_all)
                end = time.time()
                logging.info("Spending Time: " + str(round(end - start, 2)))
                # loss.append(sum((mu_mle[name] - mus[name]) ** 2))
                weight(self.weight_matrixs[name], self.Xs_tilde[name], self.data_log[name], betas_em, sigmas_em, alpha)
                
                # print(mu_mle)
                # print(sigma_matrix)
                # print(v)
                # print(s)
                # print("-"*10)
            self.betas_em[name] = betas_em
            self.sigmas_em[name] = sigmas_em

        return self.betas_em, self.sigmas_em


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
        self.radius_density, self.estimated_radius_density, self.qscore_radius_density = estimate_density(
            self.grid_points, self.distances_to_center, self.interp_func, self.betas_em)
        plot_density(self.radius_density, self.estimated_radius_density, self.qscore_radius_density,
                     amino_acid, indexes, start_rad, max_rad, gap, compared, estimated)
