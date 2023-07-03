import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal
import random
import logging
from typing import List, Dict, Tuple, Optional
from .utils import *
from .calculation import *
from copy import copy

random.seed(10)
np.random.seed(10)
logging.getLogger().setLevel(logging.INFO)

class WQEB:
    def create_data(self, 
                    group_num: int = 5,
                    group_name: list = ["A", "B", "C", "D", "E"], 
                    in_group_num: int = 6,
                    contained_ratio_data:  float = 0.1,
                    contained_ratio_beta: float = 0.1,
                    ):
        """
        Function to create data for the model.

        Params
        ----------
        group_num: int
            Number of groups.
        group_name: list
            Names of the groups.
        in_group_num: int
            Number of samples in each group.
        contained_ratio_data: float
            Contamination ratio for the data.
        contained_ratio_beta: float
            Contamination ratio for the beta values.

        Returns
        ----------
        self.data_log: dict
            Dictionary containing the generated data.
        """
        
        # hyperparameter 
        contained_group_num = int(in_group_num * contained_ratio_beta) 
        self.mus = {group_name[idx]: val for idx, val in enumerate(np.array([[-8, 8]] * group_num))}
        self.vs = {group_name[idx]: val for idx, val in enumerate([400] * group_num)}
        self.ss = {group_name[idx]: val for idx, val in enumerate([0.1] * group_num)}

        # grid points
        distances_to_center = np.repeat(np.arange(0, 0.51, 0.01), 100)
        grid_num = len(distances_to_center)
        X_tilde = [[np.concatenate((np.ones((grid_num, 1)), (-1/ 2 * distances_to_center ** 2).reshape(-1, 1)), axis=1).tolist()] * in_group_num]
        self.distances_to_center = {group_name[idx]: val for idx, val in enumerate(np.array([[distances_to_center]* in_group_num] * group_num))}
        self.Xs_tilde = {group_name[idx]: val for idx, val in enumerate(np.array(X_tilde * group_num))}
        self.lambdas = {group_name[idx]: val for idx, val in enumerate([np.ones(in_group_num)] * group_num)}
        self.weights = {group_name[idx]: val for idx, val in enumerate([np.ones((in_group_num, grid_num))] * group_num)}

        # prior
        sigmas = []
        for v, s in zip(self.vs.values(), self.ss.values()):
            sigmas.append(invgamma.rvs(a=v / 2, scale=v * s / 2, size=in_group_num))
        self.sigmas = {group_name[idx]: val for idx, val in enumerate(np.array(sigmas))}

        # beta
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
        for beta_all, sigma_all, weight_all, X_tilde_all in \
            zip(self.betas.values(), self.sigmas.values(), self.weights.values(), self.Xs_tilde.values()):
            y_tilde = []
            y_tilde_clean = []
            for beta, sigma, weight, X_tilde in zip(beta_all, sigma_all, weight_all, X_tilde_all):
                data = multivariate_normal.rvs(X_tilde @ beta, sigma * np.diag(weight))
                if contained_ratio_data > 0:
                    index = np.random.choice(np.arange(0, grid_num), int(grid_num * contained_ratio_data), replace=False)
                    data[index] = multivariate_normal.rvs(X_tilde[index, :] @ np.array([-1, 1]), sigma * np.diag(weight[index]))
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
        self.densities_data = density_mean([self.data], self.distances_to_center)[0]
        return self.data_log
    
    def read_data(
            self,
            root_map: str,
            root_pdb: str,
            atomic: Optional[str] = None,
            start_rad: float = 0,
            max_rad: float = 1,
            gap: float = 0.2,
            max_points: int = 100,
            base_num_points: int = 4,
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

        Returns
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
            atom_points, residue_names, start_rad, max_rad, gap, max_points, base_num_points)
        self.interp_func = interpolator(data, grid_size, origin)
        self.data = {key: self.interp_func(grid_points) for key, grid_points in self.grid_points.items()}
        self.data_log = {key: np.log(value + 1e-35) for key, value in self.data.items()}
        self.numbers_of_each_type = {atom: len(self.data_log[atom]) for atom in self.data_log.keys()}
        self.densities_data = density_mean([self.data], self.distances_to_center)[0]
        return self.data_log

    def paramters_initial(self)-> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        """
        Initializes the parameters for the model.

        Returns
        ----------
        Tuple containing dictionaries of initialized parameters

        mus_initial: Dict[str, np.ndarray]
            Initial values of mus
        sigmas_initial: Dict[str, np.ndarray]
            Initial values of sigmas
        weights: Dict[str, np.ndarray]
            Initial values of weights
        lambdas: Dict[str, np.ndarray]
            Initial values of lambdas
        """
            
        # Initialize mus_initial
        mus_initial = []
        for X_tilde_all, y_tilde_all in zip(self.Xs_tilde.values(), self.data_log.values()):
            mu_initial = []
            for X_tilde, y_tilde in zip(X_tilde_all, y_tilde_all):
                mu_initial.append(np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y_tilde)
            mus_initial.append(np.median(mu_initial, axis=0))
        self.mus_initial = {name: val for name, val in zip(self.data_log, mus_initial)}

        # Calculate sigmas_initial using the caluculate_sigmas function
        self.sigmas_initial = calculate_sigmas(self.Xs_tilde, self.data_log, self.mus_initial, initial=True)
        
        # Initialize weights and lambdas dictionaries
        self.weights = {name: np.ones((len(self.Xs_tilde[name]), len(self.distances_to_center[name][i]))) for name in self.Xs_tilde for i in range(len(self.Xs_tilde[name]))}
        self.lambdas = {name: np.ones(len(self.Xs_tilde[name])) for name in self.Xs_tilde}
        
        return self.mus_initial, self.sigmas_initial, self.weights, self.lambdas

    def algorithm_iter(self, 
                       max_iter: int = 3, 
                       alpha: float = 0.1, 
                       gamma: float = 0.1, 
                       tol: float = 1e-4, 
                       patience: int = 3, 
                       verbose: bool = True
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Runs the iterative algorithm to estimate parameters.

        Params
        ----------
        max_iter: int
            Maximum number of iterations.
        alpha: float
            Alpha hyperparameter.
        gamma: float
            Gamma hyperparameter.
        tol: float
            Tolerance for convergence.
        patience: int
            Number of iterations without improvement to tolerate before stopping.
        verbose: bool
            If True, display iteration progress.

        Returns
        ----------
            Tuple containing the estimated betas and the mean beta difference.
        """
        # Initialize variables
        self.mus_mle = self.mus_initial
        self.sigmas_median  = self.sigmas_initial
        least_difference = np.inf
        cur_betas = self.mus_initial
        self.beta_histories = []
        self.beta_differences_histories = []
        iter_num = 0
        for i in tqdm(range(max_iter), disable = not verbose):
            # Iterate the algorithm
            self.betas_WEB = caculate_mus_tilde(self.Xs_tilde, self.data_log, self.weights, self.lambdas, self.mus_mle)
            self.mus_mle = caculate_mus_mle(self.Xs_tilde, self.data_log, self.sigmas_median, self.weights, self.lambdas)
            self.sigmas_median = calculate_sigmas(self.Xs_tilde, self.data_log, self.betas_WEB)
            self.weights, self.lambdas = caculate_weights_and_lamdas(
                self.Xs_tilde, self.data_log, self.betas_WEB, self.sigmas_median, alpha, gamma, self.mus_mle)
            
            # Calculate the voxel values on the grid points created by the current betas 
            self.points_betas_WEB = caculate_points(self.Xs_tilde, self.betas_WEB)

            # Calculate the density of the voxel values on the grid points created by the current betas
            self.densities_betas_WEB = density_mean([self.points_betas_WEB], self.distances_to_center)[0]
            
            # Calculate the similarity between the data and the current betas
            self.similarities = caculate_similarity(self.densities_data, self.densities_betas_WEB)
            similarity_all = []
            for similarity in self.similarities.values():
                similarity_all.extend(similarity)

            # Calculate the difference between the current betas and the previous betas
            beta_differences = []
            for new_beta, cur_beta in zip(cur_betas.values(), self.betas_WEB.values()):
                beta_differences.extend((new_beta - cur_beta) ** 2)

            if verbose:
                logging.info(f"Iteration {i} finished. with difference: {np.mean(beta_differences)}")

            iter_num += 1

            # Check convergence criteria and update variables
            if np.mean(beta_differences) > tol and iter_num < (patience + 1):
                if np.mean(beta_differences) < least_difference:
                    least_difference = np.mean(beta_differences)
                    iter_num = 0
                cur_betas = self.betas_WEB
                self.beta_histories.append(self.betas_WEB)
                self.beta_differences_histories.append(np.mean(beta_differences))
            else:
                break

        self.densities_mle = caculate_density(self.distances_to_center, self.mus_mle)

        return self.betas_WEB, np.mean(beta_differences)


    def plot_data(self, max_radius: float, gap: float, root: str = False) -> None:
        """
        Plot the data densities.

        Params
        ----------
        max_radius: float
            Maximum radius for the plot.
        gap: float
            Gap between the grid points.
        root: str = None
            The root where you want to save the figure.

        Returns
        ----------
            None
        """

        # Calculate mean betas using the estimated WEB betas
        self.betas_em_mean = {name: np.mean(betas, axis=0) for name, betas in self.betas_WEB.items() if len(betas) > 0}
        
        # Calculate densities using estimated mean betas
        self.densities_em = caculate_density(self.distances_to_center, self.betas_em_mean)
        
        # Plot the densities
        plot_density(self.densities_data, 
                    [self.densities_mle, self.densities_em], 
                    max_radius, 
                    gap, 
                    ["MLE", "WEB mean"], 
                    ["blue", "red"], 
                    root=root)
