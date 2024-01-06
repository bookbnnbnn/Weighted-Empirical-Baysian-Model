import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from typing import Dict, Tuple, List

def caculate_betas_WEB(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    weight_all: np.ndarray,
    lambda_all: np.ndarray,
    mu: np.ndarray, 
) -> np.ndarray:
    """
    Calculates the mus tilde using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.
    mus: np.ndarray
        mu arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus tilde.

    """
    
    mu_tilde = []

    # Calculate mus tilde for each data point
    for X_tilde, y_tilde, weight, lambda_ in zip(X_tilde_all, y_tilde_all, weight_all, lambda_all):
        mu_tilde.append(np.linalg.inv(weight * X_tilde.T @ X_tilde + lambda_ * np.eye(2)) @ \
                        (weight * X_tilde.T @ y_tilde + lambda_ * np.eye(2) @ mu))

    return np.array(mu_tilde)


def caculate_betas_WEB_test2(
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
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.
    mus: np.ndarray
        mu arrays.

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


def calculate_sigmas(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    betas: np.ndarray,
    initial: bool = False
) -> np.ndarray:
    """
    Calculates the sigmas based on the given data and betas.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas: np.ndarray
        beta arrays.
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
    for X_tilde, y_tilde, beta in zip(X_tilde_all, y_tilde_all, betas):
        residual = (y_tilde - X_tilde @ beta)
        sigma_initial.append(np.median(residual ** 2))

    return np.array(sigma_initial)


def calculate_sigmas_MDPDE(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    betas: np.ndarray,
    weights: np.ndarray,
    alpha: float,
    initial: bool = False
) -> np.ndarray:
    """
    Calculates the sigmas based on the given data and betas.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas: np.ndarray
        beta arrays.
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


def caculate_mus_mle(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    sigma_all: np.ndarray,
    weight_all: np.ndarray,
    lambda_all: np.ndarray, 
) -> np.ndarray:
    """
    Calculates the mus MLE (maximum likelihood estimate) using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    sigmas: np.ndarray
        sigma arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus MLE.

    """

    denominators = np.array([np.eye(2)] * len(y_tilde_all))
    numerators = np.array([np.ones(2)] * len(y_tilde_all))
    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, sigma_all, weight_all, lambda_all)):          
        X_tilde, y_tilde, sigma, weight, lambda_ = elements

        # Calulate the sigma tilde matrix
        x11, x12, x22 = np.sum(X_tilde[:, 0]**2 * weight), np.sum(X_tilde[:, 0] * X_tilde[:, 1] * weight), np.sum(X_tilde[:, 1]**2 * weight)
        XWX = np.array([x11, x12, x12, x22]).reshape(2, 2)
        xy1, xy2= np.sum(X_tilde[:, 0]* y_tilde * weight), np.sum(X_tilde[:, 1] * y_tilde * weight)
        XWY = np.array([xy1, xy2])
        sigma_tilde_matrix = np.linalg.inv(XWX + lambda_ * np.eye(2))

        # Calulate the denominator and numerator to get mus MLE
        weight = lambda_ / sigma * sigma_tilde_matrix @ XWX

        denominators[idx] = weight
        numerators[idx] = weight @ np.linalg.inv(XWX) @ XWY

    return np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)


def caculate_mus_mle_test2(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    mu: np.ndarray,
    sigma_all: np.ndarray,
    sigma_matrix: np.ndarray,
    weights: np.ndarray,
    lambdas: np.ndarray, 
) -> np.ndarray:
    """
    Calculates the mus MLE (maximum likelihood estimate) using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    sigmas: np.ndarray
        sigma arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus MLE.

    """

    numerators = np.array([np.ones(2)] * len(y_tilde_all))
    denominators = np.array([np.eye(2)] * len(y_tilde_all))
    lambdas = lambdas if lambdas is not None else 1 / sigma_all
    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, sigma_all, weights, lambdas)):          
        X_tilde, y_tilde, sigma, weight, lambda_ = elements

        # Calulate the sigma tilde matrix
        x11, x12, x22 = np.sum(X_tilde[:, 0]**2 * weight), np.sum(X_tilde[:, 0] * X_tilde[:, 1] * weight), np.sum(X_tilde[:, 1]**2 * weight)
        XWX = np.array([x11, x12, x12, x22]).reshape(2, 2)
        xy1, xy2= np.sum(X_tilde[:, 0]* y_tilde * weight), np.sum(X_tilde[:, 1] * y_tilde * weight)
        XWY = np.array([xy1, xy2])

        lambda_star = lambda_ * sigma 

        sigma_matrix_inv = np.linalg.inv(sigma_matrix)
        sigma_tilde_matrix = np.linalg.inv(XWX + lambda_star * sigma_matrix_inv)

        beta_tilde = sigma_tilde_matrix @ (XWY +  lambda_star * sigma_matrix_inv @ mu)
        # Calulate the denominator and numerator to get mus MLE
        numerators[idx] = lambda_ * sigma_matrix_inv @ beta_tilde
        denominators[idx] = lambda_ * sigma_matrix_inv

    return np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)


def caculate_mus_mle_test3(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    sigma_all: np.ndarray,
    sigma_matrix: np.ndarray,
    weights: np.ndarray,
    lambdas: np.ndarray, 
) -> np.ndarray:
    """
    Calculates the mus MLE (maximum likelihood estimate) using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    sigmas: np.ndarray
        sigma arrays.
    weights: np.ndarray
        weight arrays.
    lambdas: np.ndarray
        lambda arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus MLE.

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


def caculate_betas_MDPDE(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    weight_all: np.ndarray
) -> np.ndarray:
    """
    Calculates the weighted beta values using the given data and weights.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    weights: np.ndarray
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


def caculate_mus_mle_weighted(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    sigma_all: np.ndarray,
    weight_all: np.ndarray,
    lambda_all: np.ndarray
) -> np.ndarray:
    """
    Calculates the mus MLE (maximum likelihood estimate) using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    sigmas: np.ndarray
        sigma arrays.
    weights: np.ndarray
        weight arrays.

    Returns:
    ----------
    np.ndarray
        Array containing the calculated mus MLE.

    """
    denominators = np.array([np.eye(2)] * len(y_tilde_all))
    numerators = np.array([np.ones(2)] * len(y_tilde_all))
    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, sigma_all, weight_all, lambda_all)):          
        X_tilde, y_tilde, sigma, weight, lambda_ = elements

        x11, x12, x22 = np.sum(X_tilde[:, 0]**2 * weight), np.sum(X_tilde[:, 0] * X_tilde[:, 1] * weight), np.sum(X_tilde[:, 1]**2 * weight)
        XWX = np.array([x11, x12, x12, x22]).reshape(2, 2)
        xy1, xy2= np.sum(X_tilde[:, 0]* y_tilde * weight), np.sum(X_tilde[:, 1] * y_tilde * weight)
        XWY = np.array([xy1, xy2])

        denominators[idx] = XWX / sigma * lambda_
        numerators[idx] = XWY / sigma * lambda_

    return np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)


def caculate_weights_and_lamdas(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    beta_em_all: np.ndarray,
    sigma_all: np.ndarray,
    alpha: float = None,
    gamma: float = None,
    mu_mle: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the weights and lambdas using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas_em: np.ndarray
        beta_em arrays.
    sigmas: np.ndarray
        sigma_mle arrays.
    alpha: float
        Alpha parameter value.
    gamma: float
        Gamma parameter value.
    mu_mle: np.ndarray = None
        mus mle arrays.

    Returns:
    ----------
    Tuple containing the calculated weights and lambdas.

    weights: np.ndarray
        The weight of each data point.
    lambdas: np.ndarray
        The weight of each residue
    """

    mu_mle = mu_mle if mu_mle is not None else None
    weight = np.array([np.ones(len(y_tilde_all[0]))] * len(y_tilde_all))
    lambda_ = np.ones(len(beta_em_all))

    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_em_all, sigma_all)):
        X_tilde, y_tilde, beta, sigma = elements
        
        # Calculate weight
        if alpha is not None:
            exponential_w = np.exp(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta)**2)
            # if np.sum(exponential_w > 1e10) > 0 :
                # print(sigma)
                # print(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta)**2)
                # print("----------------------------")

            # weight[idx] =  exponential_w / np.sum(exponential_w) * len(y_tilde) 
            weight[idx] =  exponential_w

        # Calculate lambda
        if (mu_mle is not None) and (gamma is not None):
            coeff_l = sigma**(-(gamma + 2) / 2)
            exponential_l = np.exp(-(gamma / (2 * sigma)) * ((beta - mu_mle).T @ (beta - mu_mle)))
            lambda_[idx] =  coeff_l * exponential_l

    return weight, lambda_


def caculate_mus_MDPDE(
    beta_all: np.ndarray,
    lambdas: np.ndarray = None, 
):
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
    # sigmas = []
    # if homo:
    #     sigma_matrices_rep = np.eye(2)
    # else:
    #     for i in range(len(sigma_all)):
    #         residual = (beta_all - mu_mle)[i].reshape(-1, 1)
    #         sigmas.append(residual @ residual.T)
    #     if lambda_all is None:
    #         sigma_matrices_rep = np.median(sigmas, axis=0)
    #     else: 
    #         sigma_matrices_rep = np.sum(sigmas, axis=0) / np.sum(lambda_all)

    #     if sigma_matrices_rep[0][0] == 0:
    #         sigma_matrices_rep[0][0] = 1e-20
    #     if sigma_matrices_rep[1][1] == 0:
    #         sigma_matrices_rep[1][1] = 1e-20

    # sigma_all = np.repeat(sigma_all, 4).reshape(len(sigma_all), 2, 2)

    # return sigma_matrices_rep, sigma_matrices_rep / sigma_all
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

    

def caculate_weights_and_lamdas_test2(
    X_tilde_all: np.ndarray,
    y_tilde_all: np.ndarray,
    beta_all: np.ndarray,
    sigma_all: np.ndarray,
    alpha: float = 0.5,
    gamma: float = 0.5,
    mu_mle: np.ndarray = None,
    sigma_matrix: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the weights and lambdas using the given data and parameters.

    Params
    ----------
    Xs_tilde: np.ndarray
        Xs tilde arrays.
    ys_tilde: np.ndarray
        ys tilde arrays.
    betas_em: np.ndarray
        beta_em arrays.
    sigmas: np.ndarray
        sigma_mle arrays.
    alpha: float
        Alpha parameter value.
    gamma: float
        Gamma parameter value.
    mu_mle: np.ndarray = None
        mus mle arrays.

    Returns:
    ----------
    Tuple containing the calculated weights and lambdas.

    weights: np.ndarray
        The weight of each data point.
    lambdas: np.ndarray
        The weight of each residue
    """

    mu_mle = mu_mle if mu_mle is not None else None
    weight = np.array([np.ones(len(y_tilde_all[0]))] * len(y_tilde_all))
    lambda_ = np.ones(len(beta_all))

    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_all, sigma_all)):
        X_tilde, y_tilde, beta, sigma = elements
        
        # Calculate weight
        weight[idx] =  np.exp(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta)**2)

        # Calculate lambda
        if (mu_mle is not None) and (gamma is not None):
            lambda_[idx] =  np.exp(-(gamma / 2) * ((beta - mu_mle).T @ np.linalg.inv(sigma_matrix) @ (beta - mu_mle))) 

    return weight, lambda_


def caculate_points(Xs_tilde: np.ndarray, betas_em: np.ndarray) -> np.ndarray:
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


def caculate_density(
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


def caculate_similarity(
    data: Dict[str, List[List[float]]],
    data_estimated: Dict[str, List[List[float]]]
) -> Dict[str, List[float]]:
    """
    Calculates the similarity scores between data densities and estimated densities.

    Params
    ----------
    densities_data: Dict[str, List[List[float]]]
        Dictionary of data densities.
    densities_estimated: Dict[str, List[List[float]]]
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