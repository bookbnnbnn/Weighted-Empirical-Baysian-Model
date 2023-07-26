import numpy as np
import pandas as pd
from scipy.stats import invgamma, multivariate_normal, chi2
import random
import logging
from typing import List, Dict, Tuple, Optional
from .utils import *
from .calculation import *
from copy import copy

logging.getLogger().setLevel(logging.INFO)

class WEB:
    def __init__(
            self, 
            start_radius: float = 0,
            max_radius: float = 1,
            gap: float = 0.2,
            ) -> None:
        
        self.start_radius = start_radius
        self.max_radius = max_radius
        self.gap = gap


    def create_data(self, 
                    group_num: int = 5,
                    group_name: list = ["A", "B", "C", "D", "E"], 
                    in_group_num: int = 6,
                    contained_ratio_data:  float = 0.1,
                    contained_ratio_beta: float = 0.1,
                    ):
        """
        Function to create data for the model.

        Parameters
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
        random.seed(0)
        np.random.seed(0)

        # hyperparameter 
        contained_group_num = int(in_group_num * contained_ratio_beta) 
        self.mus = {group_name[idx]: val for idx, val in enumerate(np.array([[-6, 6]] * group_num))}
        self.vs = {group_name[idx]: val for idx, val in enumerate([400] * group_num)}
        self.ss = {group_name[idx]: val for idx, val in enumerate([0.1] * group_num)}

        # grid points
        distances_to_center = np.repeat(np.arange(self.start_radius, self.max_radius + self.gap, self.gap), 100)
        grid_num = len(distances_to_center)
        X_tilde = [[np.concatenate((np.ones((grid_num, 1)), (-1/ 2 * distances_to_center ** 2).reshape(-1, 1)), axis=1).tolist()] * in_group_num]
        self.distances_to_center = {group_name[idx]: val for idx, val in enumerate(np.array([[distances_to_center]* in_group_num] * group_num))}
        self.Xs_tilde = {group_name[idx]: val for idx, val in enumerate(np.array(X_tilde * group_num))}
        self.lambdas = {group_name[idx]: val for idx, val in enumerate([np.ones(in_group_num)] * group_num)}
        self.weights = {group_name[idx]: val for idx, val in enumerate([np.ones((in_group_num, grid_num))] * group_num)}

        # prior
        sigma = []
        for v, s in zip(self.vs.values(), self.ss.values()):
            sigma.append(invgamma.rvs(a=v / 2, scale=v * s / 2, size=in_group_num))
        sigmas = {group_name[idx]: val for idx, val in enumerate(np.array(sigma))}

        # beta
        betas = []
        for mu, sigma_all, lambda_all in \
            zip(self.mus.values(), sigmas.values(), self.lambdas.values()):
            beta = []
            counts = contained_group_num
            for sigma, lambda_ in zip(sigma_all, lambda_all):
                if counts > 0:
                    beta.append(multivariate_normal.rvs([-8, 8], sigma * lambda_ ** (-1) * np.eye(2)))
                    counts -= 1
                else:
                    beta.append(multivariate_normal.rvs(mu, sigma * lambda_ ** (-1) * np.eye(2)))
            betas.append(np.array(beta))
        

        self.betas = {group_name[idx]: val for idx, val in enumerate(betas)}
        # data
        ys_tilde = []
        ys_tilde_clean = []
        for beta_all, sigma_all, weight_all, X_tilde_all in \
            zip(self.betas.values(), sigmas.values(), self.weights.values(), self.Xs_tilde.values()):
            y_tilde = []
            y_tilde_clean = []
            for beta, sigma, weight, X_tilde in zip(beta_all, sigma_all, weight_all, X_tilde_all):
                data = multivariate_normal.rvs(X_tilde @ beta, sigma * np.diag(weight))
                if contained_ratio_data > 0:
                    index = np.random.choice(np.arange(0, grid_num), int(grid_num * contained_ratio_data), replace=False)
                    data[index] = multivariate_normal.rvs(X_tilde[index, :] @ np.array([-4, 4]), sigma * np.diag(weight[index]))
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
            max_points: int = 100,
            base_num_points: int = 4,
    ) -> Dict[str, np.ndarray]:
        """
        Reads map and pdb files, generates grid points, interpolates data, and returns the log values of the
        interpolated data for each atom type.

        Parameters
        ----------
        root_map: str 
            Path to the map file.
        root_pdb: str 
            Path to the pdb file.
        atomic: Optional[str]
            Atom type to filter by. Defaults to None.
        max_points: int = 100
            Maximum number of points to generate at each radius. 
        base_num_points: int = 4
            Number of points to generate at the minimum radius. 

        Returns
        ----------
        Dict[str, np.ndarray]: 
            the dictionary containing the log values of the interpolated data for each atom type.
        """
        data, grid_size, origin = read_map(root_map)
        mean, std = np.mean(data), np.std(data)
        # self.A_B = mean + 10 * std, mean - std
        df_processed = read_pdb(root_pdb, atomic=atomic)
        residue_names = np.array(df_processed["residue_name"])
        atom_points = np.column_stack((df_processed.x_coord, df_processed.y_coord, df_processed.z_coord))
        self.grid_points, self.distances_to_center, self.Xs_tilde = generate_grid_points(
            atom_points, residue_names, self.start_radius, self.max_radius, self.gap, max_points, base_num_points)
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
        self.betas_initial: Dict[str, np.ndarray]
            Initial values of betas
        """

        self.mus_initial = {}
        self.sigmas_initial = {}
        self.betas_initial = {}
        for name in self.data_log:
            mu_initial = []
            for X_tilde, y_tilde in zip(self.Xs_tilde[name], self.data_log[name]):
                mu_initial.append(np.linalg.inv(X_tilde.T @ X_tilde) @ X_tilde.T @ y_tilde)
            self.mus_initial[name] = np.median(mu_initial, axis=0)
            self.sigmas_initial[name] = calculate_sigmas(self.Xs_tilde[name], self.data_log[name], self.mus_initial[name], initial=True)
            weights = np.ones((len(self.Xs_tilde[name]), len(self.distances_to_center[name][0])))
            lambdas = np.ones(len(self.Xs_tilde[name]))
            self.betas_initial[name] = caculate_betas_WEB(self.Xs_tilde[name], self.data_log[name], weights, lambdas, self.mus_initial[name])

        return self.mus_initial, self.sigmas_initial, self.betas_initial

    def WEB_iter(
            self, 
            max_iter: int = 3, 
            alpha: float = 0.1, 
            gamma: float = 0.1, 
            tol: float = 1e-4, 
            patience: int = 3, 
            verbose: int = 1, 
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Runs the iterative algorithm to estimate parameters.

        Parameters
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
        verbose: int
            If 0, show nothing. 
            If 1, display iteration progress. 
            If 2, display iteration progress and beta difference.

        Returns
        ----------
            Tuple containing the estimated betas and the mean beta difference.
        """
        # Initialize variables
        self.betas_WEB = self.betas_initial
        self.sigmas_median  = self.sigmas_initial
        self.mus_mle = self.mus_initial
        self.beta_differences_histories = {}
        self.weights = {}
        self.lambdas = {}
        for name in tqdm(self.data_log, disable = True if verbose == 0 else False):

            cur_beta = self.betas_initial[name]
            least_difference = np.inf
            beta_differences_history = []
            iter_num = 0

            for i in range(max_iter):
                # Iterate the algorithm
                self.weights[name], self.lambdas[name] = caculate_weights_and_lamdas(
                    self.Xs_tilde[name], self.data_log[name], self.betas_WEB[name], self.sigmas_median[name], alpha, gamma, self.mus_mle[name])
                self.mus_mle[name] = caculate_mus_mle(self.Xs_tilde[name], self.data_log[name], self.sigmas_median[name], self.weights[name], self.lambdas[name])
                self.betas_WEB[name] = caculate_betas_WEB(self.Xs_tilde[name], self.data_log[name], self.weights[name], self.lambdas[name], self.mus_mle[name])
                self.sigmas_median[name] = calculate_sigmas(self.Xs_tilde[name], self.data_log[name], self.betas_WEB[name])

                # Calculate the difference between the current betas and the previous betas

                beta_difference = max(np.sum((self.betas_WEB[name] - cur_beta)**2, axis=0))

                if verbose == 2:
                    logging.info(f"{name} iteration {i} finished with difference: {beta_difference}")

                iter_num += 1

                # Check convergence criteria and update variables
                if beta_difference > tol and iter_num < (patience + 1):
                    if beta_difference < least_difference:
                        least_difference = beta_difference
                        iter_num = 0
                    cur_beta = self.betas_WEB[name]
                    beta_differences_history.append(beta_difference)
                else:
                    self.betas_WEB[name] = cur_beta
                    break

            self.beta_differences_histories[name] = beta_differences_history
            
        self.densities_mle = caculate_density(self.distances_to_center, self.mus_mle)

        return self.betas_WEB, self.beta_differences_histories
    

    def WLR_iter(
            self, 
            max_iter: int = 3, 
            alpha: float = 0.1, 
            gamma: float = 0.1, 
            tol: float = 1e-4, 
            patience: int = 3, 
            verbose: int = 1
    ) -> Tuple[Dict[str, np.ndarray], float]:
        """
        Runs the iterative algorithm to estimate parameters.

        Parameters
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
        verbose: int
            If 0, show nothing. 
            If 1, display iteration progress. 
            If 2, display iteration progress and beta difference.

        Returns
        ----------
            Tuple containing the estimated betas and the mean beta difference.
        """

        # Initialize variables
        self.betas_WLR = self.betas_initial
        sigmas_median = self.sigmas_initial
        self.mus_mle_weighted = self.mus_initial
        weights = {}
        lambdas = {}
        for name in tqdm(self.data_log, disable = True if verbose == 0 else False):

            cur_beta = self.betas_initial[name]
            least_difference = np.inf
            iter_num = 0

            for i in range(max_iter):
                # Iterate the algorithm
                weights[name], lambdas[name] = caculate_weights_and_lamdas(
                    self.Xs_tilde[name], self.data_log[name], self.betas_WLR[name], sigmas_median[name], alpha, gamma, self.mus_mle_weighted[name])
                self.mus_mle_weighted[name] = caculate_mus_mle_weighted(self.Xs_tilde[name], self.data_log[name], sigmas_median[name], weights[name], lambdas[name])
                self.betas_WLR[name] = caculate_betas_WLR(self.Xs_tilde[name], self.data_log[name], weights[name])
                sigmas_median[name] = calculate_sigmas(self.Xs_tilde[name], self.data_log[name], self.betas_WLR[name])

                # Calculate the difference between the current betas and the previous betas

                beta_difference = max(np.sum((self.betas_WLR[name] - cur_beta)**2, axis=0))

                if verbose == 2:
                    logging.info(f"{name} iteration {i} finished with difference: {beta_difference}")

                iter_num += 1

                # Check convergence criteria and update variables
                if beta_difference > tol and iter_num < (patience + 1):
                    if beta_difference < least_difference:
                        least_difference = beta_difference
                        iter_num = 0
                    cur_beta = self.betas_WLR[name]
                else:
                    self.betas_WLR[name] = cur_beta
                    break
            
        # self.densities_mle = caculate_density(self.distances_to_center, self.mus_mle)

        return self.betas_WLR


    def fitted_densities_plot(self, root: Optional[str] = None):
        """
        Plot the fitted density distributions of data points using the WEB method.

        Parameters
        ----------
        root : str, optional
            The file path to save the plot, by default None.

        Returns
        -------
        None
        """

        fitted_densities = caculate_density(self.distances_to_center, self.betas_WEB, separated=True)
        separated_densities = {f"{key}_{num + 1}": [density] for key, value in self.densities_data.items() for num, density in enumerate(value)}
        separated_fitted_densities = {f"{key}_{num + 1}": density for key, value in fitted_densities.items() for num, density in enumerate(value)}

        legend = {
        "loc": "lower left",
        "bbox_to_anchor": (0.2, 0.025),
        "bbox_transform": plt.gcf().transFigure,
        "ncol": 2
        }
        plot_row = len(separated_densities) // 5 if len(separated_densities) % 5 == 0 else len(separated_densities) // 5 + 1

        sub_plots(
            fitted_density_plot, 
            separated_densities, 
            x_label = "Radius",
            y_label = "Voxel Value", 
            fontsize = 25,
            plot_dim = (plot_row , 5),
            figsize = (25, plot_row * 4),
            sharex=True, 
            sharey=True,
            legend=legend,
            root=root,
            estimated_densities=[separated_fitted_densities],
            start_radius=self.start_radius,
            max_radius=self.max_radius, 
            gap=self.gap,
            labels=["WEB"],
            colors=["blue"],
            )


    def representative_densities_plot(self, 
            root: Optional[str] = None, 
            estimators: List[str] = ["WEB MLE", "WEB Mean-E", "Map Mean", "WEB-E Mean", "WEB-E W-Mean"]):
        """
        Plot the representative density distributions of data points using different estimators.

        Parameters
        ----------
        root : str, optional
            The file path to save the plot, by default None.
        estimators : List[str], optional
            List of estimator names to plot, by default ["WEB MLE", "WEB Mean-E", "Map Mean", "WEB-E Mean", "WEB-E W-Mean"].

        Returns
        -------
        None
        """

        estimated_densities = []

        if "WEB MLE" in estimators:
            estimated_densities.append(self.densities_mle)

        # Calculate mean betas using the estimated WEB betas and do exponential transformation
        if "WEB Mean-E" in estimators:
            self.betas_EWEB = {name: np.mean(betas, axis=0) for name, betas in self.betas_WEB.items() if len(betas) > 0}
            self.densities_EWEB_Mean = caculate_density(self.distances_to_center, self.betas_EWEB)
            estimated_densities.append(self.densities_EWEB_Mean)

        # Calculate mean densities of denstiy maps
        if "Map Mean" in estimators:
            self.densities_Mean = {name: np.mean(densities, axis=0) for name, densities in self.densities_data.items()}
            estimated_densities.append(self.densities_Mean)

        # Calculate mean of estimated WEB betas after exponential transformation
        if "WEB-E Mean" in estimators:
            self.densities_WEB = caculate_density(self.distances_to_center, self.betas_WEB, seperated=True)
            self.densities_WEBE_Mean = {name: np.mean(density, axis=0) for name, density in self.densities_WEB.items()}
            estimated_densities.append(self.densities_WEBE_Mean)

        # Calculate mean of estimated WEB betas after exponential transformation with weights
        if "WEB-E W-Mean" in estimators:
            self.densities_WEB = caculate_density(self.distances_to_center, self.betas_WEB, seperated=True)
            self.densities_WEBE_WMean = {name: np.sum(self.lambdas[name].reshape(-1, 1) * self.densities_WEB[name], axis=0) / np.sum(self.lambdas[name]) for name, density in self.densities_WEB.items()}
            estimated_densities.append(self.densities_WEBE_WMean)

        # Calculate empirical Bayes estimates
        if "EB" in estimators:
            self.mus_EB = {name: caculate_mus_mle(self.Xs_tilde[name], self.data_log[name], self.sigmas_median[name], 
                                                    np.ones((len(self.Xs_tilde[name]), len(self.distances_to_center[name][0]))), np.ones(len(self.Xs_tilde[name]))) 
                                                    for name in self.data_log}
            self.densities_EB = caculate_density(self.distances_to_center, self.mus_EB)
            estimated_densities.append(self.densities_EB)

        colors = ["blue", "black", "green", "purple", "gray", "red"]

        legend = {
        "loc": "lower center",
        "bbox_to_anchor": (1.5, 0.1),
        "bbox_transform": plt.gcf().transFigure,
        "ncol": len(estimators)
        }

        sub_plots(
            fitted_density_plot, 
            self.densities_data, 
            x_label = "Radius",
            y_label = "Voxel Value", 
            fontsize = 25,
            plot_dim = (4, 5),
            figsize = (25, 16),
            sharex=True, 
            sharey=True,
            legend=legend,
            root=root,
            estimated_densities=estimated_densities,
            start_radius=self.start_radius,
            max_radius=self.max_radius, 
            gap=self.gap,
            labels=estimators,
            colors=colors[: len(estimators)],
            )
        
        
    def find_outliers(self, prob: float = 0.9973) -> Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]:
        """
        Find outliers in the data using the weighted empirical Bayesian method.

        Parameters
        ----------
        prob : float, optional
            The probability value for the chi-square distribution to determine outliers, by default 0.9973.

        Returns
        -------
        Tuple[Dict[str, List[int]], Dict[str, np.ndarray]]
            A tuple containing dictionaries of outliers and statistical distances for each amino acid.
        """

        self.outliers = {}
        self.densities_outliers = {}
        self.statistic_distances0 = {}
        self.statistic_distances1 = {}
        self.statistic_distances = {}
        self.sigmas = {}
        self.margin = chi2.ppf(prob, df=2)
        for name in self.betas_WEB:
            self.statistic_distances0[name] = np.array([]) 
            self.statistic_distances1[name] = np.array([]) 
            self.statistic_distances[name] = np.array([]) 
            mean = self.mus_mle[name]
            betas = self.betas_WEB[name]
            lambdas = self.lambdas[name]
            sigma0 = np.sum([lambdas[i] * (betas - mean)[i][0] ** 2 for i in range(len(betas))]) / sum(lambdas)
            sigma1 = np.sum([lambdas[i] * (betas - mean)[i][1] ** 2 for i in range(len(betas))]) / sum(lambdas)
            sigma01 = np.sum([lambdas[i] * (betas - mean)[i][0] * (betas - mean)[i][1] for i in range(len(betas))]) / sum(lambdas)
            self.sigmas[name] = np.array([sigma0, sigma01, sigma01, sigma1]).reshape(2, 2)

            for i in range(len(betas)):
                statistic_distance0 = (betas - mean)[i][0] ** 2 / sigma0
                statistic_distance1 = (betas - mean)[i][1] ** 2 / sigma1
                
                self.statistic_distances0[name] = np.append(self.statistic_distances0[name], statistic_distance0)
                self.statistic_distances1[name] = np.append(self.statistic_distances1[name], statistic_distance1)

                statistic_distance = ((betas - mean)[i]).T @np.linalg.inv(self.sigmas[name]) @ (betas - mean)[i]
                self.statistic_distances[name] = np.append(self.statistic_distances[name], statistic_distance)

                if (statistic_distance) > self.margin:
                    if name not in self.outliers:
                        self.outliers[name] = []
                        self.densities_outliers[name] = []
                    self.outliers[name].append(i)
                    self.densities_outliers[name].append(self.densities_data[name][i])
        
        return self.outliers, self.statistic_distances
    

    def distances_hist(self, root: Optional[str] = None) -> None:
        """
        Plot the histogram of log statistical distances for weighted empirical Bayesian analysis.

        Parameters
        ----------
        root : Optional[str], optional
            The root where you want to save the figure, by default None.

        Returns
        -------
        None
        """

        sub_plots(
            distance_hist, 
            self.outliers, 
            x_label = "Log Statistical Distance",
            y_label = "Count", 
            sharex=True, 
            sharey=True,
            root=root,
            statistic_distances = self.statistic_distances, 
            margin = self.margin
            )
        
    def confidence_regions_plot(self, root: Optional[str] = None):
        """
        Plot confidence regions for normal and outlier data points.

        Parameters
        ----------
        root : str, optional
            The file path to save the plot, by default None.

        Returns
        -------
        None
        """

        point_n = Line2D([0], [0], label='Normal', marker='o', markersize=10, markeredgecolor='blue', markerfacecolor='blue', linestyle='')
        point_o = Line2D([0], [0], label='Outliers', marker='o', markersize=10, markeredgecolor='red', markerfacecolor='#ff7f0e', linestyle='')
        green_patch = Patch(color='g', label='Confidence Region', alpha=0.5)
        legend = {
            "handles": [point_n, point_o, green_patch],
            "loc": "lower center",
            "bbox_to_anchor": (1, 0.1),
            "bbox_transform": plt.gcf().transFigure,
            "ncol": 3  
        }

        sub_plots(
            confidence_region_plot, 
            self.outliers, 
            x_label = "Intercept",
            y_label = "Slope", 
            legend = legend,
            root=root,
            statistic_distances = self.statistic_distances, 
            betas_WEB = self.betas_WEB, 
            sigmas = self.sigmas,
            mus_mle = self.mus_mle, 
            margin = self.margin
            )
    
    def outliers_density_plot(self, root: Optional[str] = None):
        """
        Plot the density distributions of normal and outlier data points.

        Parameters
        ----------
        root : str, optional
            The file path to save the plot, by default None.

        Returns
        -------
        None
        """

        legend = {
        "loc": "lower center",
        "bbox_to_anchor": (1.5, 0.1),
        "bbox_transform": plt.gcf().transFigure,
        "ncol": 3
        }

        sub_plots(
            outliers_density_plot, 
            self.densities_data, 
            x_label = "Radius",
            y_label = "Voxel Value", 
            fontsize = 25,
            plot_dim = (4, 5),
            figsize = (25, 16),
            sharex=True, 
            sharey=True,
            legend = legend,
            root=root,
            densities_mle = self.densities_mle,
            densities_outliers = self.densities_outliers,
            start_radius=self.start_radius,
            max_radius=self.max_radius, 
            gap=self.gap
            )
        
    def densities_plot(self, root: Optional[str] = None):
        """
        Plot the density distributions of data points.

        Parameters
        ----------
        root : str, optional
            The file path to save the plot, by default None.

        Returns
        -------
        None
        """

        sub_plots(
            density_plot, 
            self.densities_data, 
            x_label = "Radius",
            y_label = "Voxel Value", 
            fontsize = 25,
            plot_dim = (4, 5),
            figsize = (25, 16),
            sharex=True, 
            sharey=True,
            root=root,
            start_radius=self.start_radius,
            max_radius=self.max_radius, 
            gap=self.gap
            )

    

