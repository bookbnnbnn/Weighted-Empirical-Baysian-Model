import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from typing import List, Dict, Tuple, Optional
from QEB.utils import *

random.seed(0)
np.random.seed(0)


class QEB:
    def create_data(
            self,
            mu_0: int = 2,
            sigma_0: int = 0.5,
            numbers_of_each_type: Dict[str, int] = {"A": 3, "B": 2},
            adjustment: Dict[str, List[int]] = {"A": [1] * 3, "B": [1.2] * 2},
            distances_to_center: Dict[str, List[np.ndarray]] = {"A": [np.repeat(np.array([1, 2, 3]), 1000)]*3,
                                                                "B": [np.repeat(np.array([1, 2, 3]), 1000)]*2},
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
        distances_to_center:  Dict[str, List[np.ndarray]]
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
        self.distances_to_center = distances_to_center
        self.variances_of_error = variances_of_error
        self.types_of_proteins = len(numbers_of_each_type)

        self.beta = {key: np.abs(np.random.normal(self.mu_0, self.sigma_0))
                     for key in numbers_of_each_type.keys()}
        self.data_log = {}
        for type_j, sample_size in numbers_of_each_type.items():
            beta = self.beta[type_j]
            center_distances = distances_to_center[type_j]
            # epsilon = np.array([np.random.normal(0, variance, center_distances.shape)
            #                    for variance in variances_of_error[type_j]])
            variance = variances_of_error[type_j]
            A_ij = adjustment[type_j]
            A_ij_tilde = -3 / 2 * np.log(2 * np.pi * 1 / beta) + np.log(A_ij)
            data_list = []
            for i in range(sample_size):
                X_k = - 1 / 2 * center_distances[i]**2
                epsilon = np.random.normal(0, variance[i])
                data_list.append(A_ij_tilde[i] + beta * X_k + epsilon)
            self.data_log[type_j] = np.array(data_list)
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
        self.grid_points, self.distances_to_center = generate_grid_points(
            df_processed, start_rad, max_rad, gap, max_points, base_num_points, max_iter)
        self.interp_func = interpolator(data, grid_size, origin)
        self.data = {key: self.interp_func(grid_points) for key, grid_points in self.grid_points.items()}
        self.data_log = {key: np.log(value + 1e-35) for key, value in self.data.items()}
        self.numbers_of_each_type = {atom: len(self.data_log[atom]) for atom in self.data_log.keys()}
        return self.data_log

    def caculate_constants(self) -> Tuple[Dict[str, np.ndarray]]:
        """
        Estimate `A_ij_tilde` by simple linear regression solved by OLS.
        Estimate `epsilon_ij` by MSE.
        Caculate the constants of `B_j`, `C_j`, `D_j` `sigma_j_bar` before caculate log likelihood function.

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
            data_log_arr = self.data_log[type_j]
            center_distances_list = self.distances_to_center[type_j]
            for i in range(sample_size):
                data_log = data_log_arr[i]
                center_distances = center_distances_list[i]
                X_k = - 1 / 2 * center_distances**2
                cov = np.cov(X_k, data_log)[0, 1]
                var = np.var(X_k)
                beta_ols = cov / var
                A_ij = data_log.mean() - beta_ols * X_k.mean()
                variance = sum((data_log - (A_ij + X_k * beta_ols))
                               ** 2) / (data_log.shape[0] - 1)
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

        return self.estimated_A_ij_tilde, self.B_j, self.C_j, self.D_j

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
        return -sum(-1 / 2 * (np.log(E_j)) - (sigma_0 * (4 * B_j * \
                    D_j - C_j**2) - 2 * mu_0 * C_j - 2 * B_j - 2 * mu_0**2 * D_j) / (2 * E_j))

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
                            (2, 0.5), method='Nelder-Mead', bounds=((None, None), (0, None)))
        self.mu_0_hat, self.sigma_0_hat = self.res.x
        return self.mu_0_hat, self.sigma_0_hat

    def estimate_beta_1j_by_Bayse(self) -> Dict[str, np.ndarray]:
        """
        Estimate `beta_1j` by bayes' estimator.

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

    def calculate_qeb_score(self) -> Dict[str, List[float]]:
        """
        Calculates the Q-score for each amino acid residue in the protein structure.
        
        Return
        ----------
        Dict[str, List[float]
            A dictionary of Q-scores for each amino acid residue.
        """
        self.qeb_score = {key: [] for key in self.data.keys()}
        self.estimated_data = {key: [] for key in self.data.keys()}
        for amino_acid, densities in self.data.items():
            # Calculate estimated densities from empirical baysian model
            X_k = - 1 / 2 * np.array(self.distances_to_center[amino_acid])**2
            A_ij_tilde = self.estimated_A_ij_tilde[amino_acid].reshape(-1, 1)
            beta_1j = self.beta_1j_hat[amino_acid].reshape(-1, 1)
            estimated_densities = np.exp(A_ij_tilde + X_k * beta_1j)
            self.estimated_data[amino_acid] = estimated_densities
            for num in range(estimated_densities.shape[0]):
                # Get density and estimated_density at current index
                density = densities[num]
                estimated_density = estimated_densities[num]
                # Calculate dot product and distance between density and estimated_density
                dot = np.dot(density - density.mean(), estimated_density - estimated_density.mean())
                distance = (np.linalg.norm(density - density.mean()) * np.linalg.norm(estimated_density - estimated_density.mean()))
                self.qeb_score[amino_acid].append(dot / distance)
        return self.qeb_score


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
            self.grid_points, self.distances_to_center, self.interp_func, self.beta_1j_hat, self.estimated_A_ij_tilde)
        plot_density(self.radius_density, self.estimated_radius_density, self.qscore_radius_density, self.A_B ,
                     amino_acid, indexes, start_rad, max_rad, gap, compared, estimated)
