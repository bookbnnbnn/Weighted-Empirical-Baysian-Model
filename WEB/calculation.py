import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple, List




def calculate_betas_WEB(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    weights: np.ndarray,
    lambdas: np.ndarray,
    mu: np.ndarray, 
    sigmas: np.ndarray,
    sigma_matrix: np.ndarray,
) -> np.ndarray:
    """
    Calculates the mus tilde using the given data and parameters.

    Params
    ----------
    X_tilde_all: np.ndarray
        X_tilde arrays.
    y_tilde_all: np.ndarray
        y_tilde arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.
    mu: np.ndarray
        mu arrays.
    sigmas: np.ndarray
        sigma arrays.
    sigma_matrix: np.ndarray
        sigma_matrix arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus tilde.
    """
    
    beta_hat = []

    # Calculate mus tilde for each data point
    for X_tilde, y_tilde, weight, lambda_, sigma in zip(X_tilde_all, y_tilde_all, weights, lambdas, sigmas):
        lambda_star = lambda_ * sigma
        beta_hat.append(np.linalg.inv(weight * X_tilde.T @ X_tilde + lambda_star * np.linalg.inv(sigma_matrix)) @ \
                        (weight * X_tilde.T @ y_tilde + lambda_star * np.linalg.inv(sigma_matrix) @ mu))

    return np.array(beta_hat)


def calculate_sigmas_MDPDE(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    betas: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    initial: bool = False
) -> np.ndarray:
    """
    Calculates the sigmas based on the given data, betas, and weights using the MDPDE method.

    Params
    ----------
    X_tilde_all: np.ndarray
        X_tilde arrays.
    y_tilde_all: np.ndarray
        y_tilde arrays.
    betas: np.ndarray
        beta arrays.
    weights: np.ndarray
        weight arrays.
    alpha: float
        Alpha hyperparameter for MDPDE.
    initial: bool = False
        Flag indicating whether initial sigmas are being calculated.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated sigmas.
    """

    sigma_initial = []

    if initial:
        betas = [betas] * len(y_tilde_all)

    # Calculate sigmas for each data point
    for X_tilde, y_tilde, beta, weight in zip(X_tilde_all, y_tilde_all, betas, weights):
        residual = (y_tilde - X_tilde @ beta)
        nominator = np.sum(weight * residual ** 2)
        denominator = np.sum(weight - alpha / (1 + alpha) ** (3 / 2))
        sigma_initial.append(nominator / denominator)

    return np.array(sigma_initial)



def calculate_mus_mle(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    sigma_all: np.ndarray,
    sigma_matrix: np.ndarray,
    weights: np.ndarray,
    lambdas: np.ndarray, 
) -> np.ndarray:
    """
    Calculates the MLE (maximum likelihood estimate) of mus using the given data and parameters.

    Params
    ----------
    X_tilde_all: np.ndarray
        X_tilde arrays.
    y_tilde_all: np.ndarray
        y_tilde arrays.
    sigma_all: np.ndarray
        sigma arrays.
    sigma_matrix: np.ndarray
        sigma_matrix array.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated MLE of mus.
    """

    numerators = np.array([np.ones(2)] * len(y_tilde_all))
    denominators = np.array([np.eye(2)] * len(y_tilde_all))
    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, sigma_all, weights, lambdas)):          
        X_tilde, y_tilde, sigma, weight, lambda_ = elements
        lambda_star = lambda_ * sigma 
        sigma_matrix_inv = np.linalg.inv(sigma_matrix)

        # Calulate the sigma tilde matrix
        x11, x12, x22 = np.sum(X_tilde[:, 0]**2 * weight), np.sum(X_tilde[:, 0] * X_tilde[:, 1] * weight), np.sum(X_tilde[:, 1]**2 * weight)
        XWX = np.array([x11, x12, x12, x22]).reshape(2, 2)
        xy1, xy2= np.sum(X_tilde[:, 0]* y_tilde * weight), np.sum(X_tilde[:, 1] * y_tilde * weight)
        XWY = np.array([xy1, xy2])
        sigma_tilde_matrix = np.linalg.inv(XWX + lambda_star * sigma_matrix_inv)

        numerators[idx] = lambda_ * sigma_matrix_inv @ sigma_tilde_matrix @ XWY 
        denominators[idx] = lambda_ * sigma_matrix_inv @ (np.eye(len(sigma_matrix)) - lambda_star * sigma_tilde_matrix @ sigma_matrix_inv)

    return np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)


def calculate_betas_MDPDE(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    weight_all: np.ndarray
) -> np.ndarray:
    """
    Calculates the weighted beta values using the given data and weights.

    Params
    ----------
    X_tilde_all: np.ndarray
        X_tilde arrays.
    y_tilde_all: np.ndarray
        y_tilde arrays.
    weight_all: np.ndarray
        weight arrays.

    Returns
    ----------
    np.ndarray
        Array containing the calculated weighted beta values.
    """

    beta_tilde = []

    # Calculate weighted beta for each data point
    for weight, X_tilde, y_tilde in zip(weight_all, X_tilde_all, y_tilde_all):
        beta_tilde.append(np.linalg.inv(weight * X_tilde.T @ X_tilde) @ (weight * X_tilde.T @ y_tilde))

    return np.array(beta_tilde)



def calculate_weights(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    beta_em_all: np.ndarray,
    sigma_all: np.ndarray,
    alpha: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the weights using the given data and parameters.

    Params
    ----------
    X_tilde_all: np.ndarray
        X_tilde arrays.
    y_tilde_all: np.ndarray
        y_tilde arrays.
    beta_em_all: np.ndarray
        beta_em arrays.
    sigma_all: np.ndarray
        sigma_mle arrays.
    alpha: float
        Alpha parameter value.

    Returns:
    ----------
    Tuple containing the calculated weights.

    weights: np.ndarray
        The weight of each data point.
    """

    weight = np.array([np.ones(len(y_tilde_all[0]))] * len(y_tilde_all))

    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_em_all, sigma_all)):
        X_tilde, y_tilde, beta, sigma = elements
        
        # Calculate weight
        if alpha is not None:
            exponential_w = np.exp(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta)**2)
            weight[idx] =  exponential_w

    return weight



def calculate_mus_MDPDE(
    beta_all: np.ndarray,
    lambdas: np.ndarray = None, 
):
    """
    Calculates the mus using the given beta values and lambdas.

    Params
    ----------
    beta_all: np.ndarray
        beta arrays.
    lambdas: np.ndarray
        lambda arrays, optional.

    Returns
    ----------
    np.ndarray
        Array containing the calculated mus.

    """
    numerator = np.sum([lambda_ * beta for lambda_, beta in zip(lambdas, beta_all)], axis=0)
    denominator = np.sum(lambdas)

    return numerator / denominator


def calculate_sigma_matrix_MDPDE(
    beta_all: np.ndarray,
    mu_mle: np.ndarray,
    lambdas: np.ndarray = None, 
    gamma: float = None,
    initial: bool = False, 
    params: tuple = None
):
    """
    Calculates the sigma matrix using the given beta values, mu MLE, and lambdas.

    Params
    ----------
    beta_all: np.ndarray
        beta arrays.
    mu_mle: np.ndarray
        mu MLE arrays.
    lambdas: np.ndarray, optional
        lambda arrays.
    gamma: float, optional
        Gamma parameter value.
    initial: bool, optional
        Flag indicating whether initial sigma matrix is being calculated.
    params: tuple, optional
        Lower and upper bounds for selecting data points.

    Returns
    ----------
    np.ndarray
        Calculated sigma matrix.

    """
    residuals = beta_all - mu_mle
    lower, upper = params if params is not None else (0.4, 0.6)
    if initial:
        r_square = np.sum(residuals ** 2, axis=1)
    
        selected_index = np.argsort(r_square)[int(len(r_square) * lower): int(len(r_square) * upper)]
        selected_beta = beta_all[selected_index]

        sigma_matrix = np.zeros((2, 2))
        for selected_residual in (selected_beta - mu_mle):
            selected_residual = selected_residual.reshape(-1, 1)
            sigma_matrix += selected_residual @ selected_residual.T / len(selected_beta)
        
        return sigma_matrix
    numerators = []
    for residual, lambda_ in zip(residuals, lambdas):
        residual = residual.reshape(-1, 1)
        numerators.append(lambda_ * residual @ residual.T)
    numerator = np.sum(numerators, axis=0)
    denominator = np.sum(lambdas - gamma / ((1 + gamma) ** (3/2)))

    return numerator / denominator
    

def calculate_lamdas(
    beta_all: np.ndarray,
    gamma: float = 0.5,
    mu_mle: np.ndarray = None,
    sigma_matrix: np.ndarray = None,
) -> np.ndarray:
    """
    Calculates the lambdas using the given data and parameters.

    Params
    ----------
    beta_all: np.ndarray
        beta_em arrays.
    gamma: float, optional
        Gamma parameter value.
    mu_mle: np.ndarray, optional
        mus mle arrays.
    sigma_matrix: np.ndarray, optional
        Sigma matrix.

    Returns
    ----------
    np.ndarray
        The weight of each residue.
    """

    mu_mle = mu_mle if mu_mle is not None else None
    lambda_ = np.ones(len(beta_all))

    for idx, beta in enumerate(beta_all):

        # Calculate lambda
        if (mu_mle is not None) and (gamma is not None):
            lambda_[idx] =  np.exp(-(gamma / 2) * ((beta - mu_mle).T @ np.linalg.inv(sigma_matrix) @ (beta - mu_mle))) 

    return lambda_



def calculate_points(Xs_tilde: np.ndarray, betas_em: np.ndarray) -> np.ndarray:
    """
    Calculates the points using the given Xs tilde and beta_em.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    betas_em:  np.ndarray
        beta_em arrays.

    Returns
    ----------
    np.ndarray
        Array containing the calculated points.

    """

    points = {name: None for name in Xs_tilde}

    for name, X_tilde_all, beta_em_all in zip(Xs_tilde, Xs_tilde.values(), betas_em.values()):
        density = np.array([np.ones(len(X_tilde_all[0]))] * len(X_tilde_all))

        # Calculate density for each Xs tilde and beta_em
        for idx, elements in enumerate(zip(X_tilde_all, beta_em_all)):
            X_tilde, beta_em = elements
            density[idx] = np.exp(X_tilde @ beta_em)

        points[name] = density
            
    return points


def calculate_density(
    distances_to_center: np.ndarray,
    betas: np.ndarray,
    separated: bool = False
) -> np.ndarray:
    """
    Calculates the densities using the given distances to center and betas.

    Params
    ----------
    distances_to_center: np.ndarray
        distances to center arrays.
    betas: np.ndarray
        beta arrays.
    separated: bool = False
        Flag indicating whether densities are being calculated for each group seperately.

    Returns
    ----------
    np.ndarray
        Array containing the calculated densities.

    """

    densities = {name: None for name in distances_to_center}

    # Calculate densities for each group
    for name in distances_to_center:
        distance_to_center = np.unique(distances_to_center[name][0])
        X_tilde = np.array([np.ones(len(distance_to_center)), -1 / 2 * distance_to_center**2])
        if separated:
            density = []
            for i in range(len(betas[name])):
                density.append(np.exp(X_tilde.T @ betas[name][i]))
            densities[name] = np.array(density)
        else:
            densities[name] = np.exp(X_tilde.T @ betas[name])

    return densities


def calculate_similarity(
    data: Dict[str, List[List[float]]],
    data_estimated: Dict[str, List[List[float]]]
) -> Dict[str, List[float]]:
    """
    Calculates the similarity scores between data densities and estimated densities.

    Params
    ----------
    data: Dict[str, List[List[float]]]
        Dictionary of data densities.
    data_estimated: Dict[str, List[List[float]]]
        Dictionary of estimated densities.

    Returns:
    ----------
    Dict[str, List[float]]
        Dictionary containing the similarity scores for each group.

    """

    qscores_all = {}

    for name in data:

        qscores = []
        # Calculate similarity score for each data and estimated density pair
        for datum, estimated_datum in zip(data[name], data_estimated[name]):
            numerator = np.dot(datum - datum.mean(), estimated_datum - estimated_datum.mean())
            denominator = np.sqrt(sum((datum - datum.mean())**2)) * np.sqrt(sum((estimated_datum - estimated_datum.mean())**2))
            qscores.append(numerator / denominator)

        qscores_all[name] = qscores

    return qscores_all


def solve(beta):
    beta_0, beta_1 = beta
    tau = 1 / beta_1
    A = np.exp(beta_0 + 3 / 2 * np.log(beta_1))
    return tau, A