import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from numba import njit
from typing import Dict, Tuple, List

def caculate_mus_tilde(
    Xs_tilde: Dict[str, np.ndarray],
    ys_tilde: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    lambdas: Dict[str, np.ndarray],
    mus: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculates the mus tilde using the given data and parameters.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    ys_tilde: Dict[str, np.ndarray]
        Dictionary of ys tilde arrays.
    weights: Dict[str, np.ndarray]
        Dictionary of weight arrays.
    lambdas: Dict[str, np.ndarray]
        Dictionary of lambda arrays.
    mus: Dict[str, np.ndarray]
        Dictionary of mu arrays.

    Returns:
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated mus tilde.

    """
    
    mus_tilde = []

    # Iterate over the data and parameters
    for X_tilde_all, y_tilde_all, mu, weight_all, lambda_all in \
        zip(Xs_tilde.values(), ys_tilde.values(), mus.values(), weights.values(), lambdas.values()):

        mu_tilde = []

        # Calculate mus tilde for each data point
        for weight, X_tilde, y_tilde, lambda_ in zip(weight_all, X_tilde_all, y_tilde_all, lambda_all):
            mu_tilde.append(np.linalg.inv(weight * X_tilde.T @ X_tilde + lambda_ * np.eye(2)) @ \
                            (weight * X_tilde.T @ y_tilde + lambda_ * np.eye(2) @ mu))
        
        mus_tilde.append(np.array(mu_tilde))

    return {name: np.array(val) for name, val in zip(ys_tilde, mus_tilde)}


def calculate_sigmas(
    Xs_tilde: Dict[str, np.ndarray],
    ys_tilde: Dict[str, np.ndarray],
    betas: Dict[str, np.ndarray],
    initial: bool = False
) -> Dict[str, np.ndarray]:
    """
    Calculates the sigmas based on the given data and betas.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    ys_tilde: Dict[str, np.ndarray]
        Dictionary of ys tilde arrays.
    betas: Dict[str, np.ndarray]
        Dictionary of beta arrays.
    initial: bool = False
        Flag indicating whether initial sigmas are being calculated.

    Returns:
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated sigmas.

    """

    sigmas_initial = []

    # Iterate over the data and betas
    for X_tilde_all, y_tilde_all, betas in zip(Xs_tilde.values(), ys_tilde.values(), betas.values()):

        sigma_initial = []

        if initial:
            betas = [betas] * len(y_tilde_all)

        # Calculate sigmas for each data point
        for X_tilde, y_tilde, beta in zip(X_tilde_all, y_tilde_all, betas):
            residual = (y_tilde - X_tilde @ beta)
            sigma_initial.append(np.median(residual ** 2))

        sigmas_initial.append(sigma_initial)

    return {name: np.array(val) for name, val in zip(ys_tilde, sigmas_initial)}


def caculate_mus_mle(Xs_tilde: Dict[str, np.ndarray],
    ys_tilde: Dict[str, np.ndarray],
    sigmas: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray],
    lambdas: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculates the mus MLE (maximum likelihood estimate) using the given data and parameters.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    ys_tilde: Dict[str, np.ndarray]
        Dictionary of ys tilde arrays.
    sigmas: Dict[str, np.ndarray]
        Dictionary of sigma arrays.
    weights: Dict[str, np.ndarray]
        Dictionary of weight arrays.
    lambdas: Dict[str, np.ndarray]
        Dictionary of lambda arrays.

    Returns:
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated mus MLE.

    """
    mus_mle = {name: None for name in ys_tilde}
    for name, X_tilde_all, y_tilde_all, sigma_all, weight_all, lambda_all in \
        zip(ys_tilde, Xs_tilde.values(), ys_tilde.values(), sigmas.values(), weights.values(), lambdas.values()):
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
            denominator = lambda_ / sigma * (np.eye(2) - sigma_tilde_matrix * lambda_)
            numerator = lambda_ / sigma * sigma_tilde_matrix @ XWY
            denominators[idx] = denominator
            numerators[idx] = numerator
        mus_mle[name] = np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)
    return mus_mle


def caculate_weighted_beta(
    Xs_tilde: Dict[str, np.ndarray],
    ys_tilde: Dict[str, np.ndarray],
    weights: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculates the weighted beta values using the given data and weights.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    ys_tilde: Dict[str, np.ndarray]
        Dictionary of ys tilde arrays.
    weights: Dict[str, np.ndarray]
        Dictionary of weight arrays.

    Returns
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated weighted beta values.

    """
    
    betas_tilde = []

    # Iterate over the data and weights
    for X_tilde_all, y_tilde_all, weight_all in zip(Xs_tilde.values(), ys_tilde.values(), weights.values()):

        beta_tilde = []

        # Calculate weighted beta for each data point
        for weight, X_tilde, y_tilde in zip(weight_all, X_tilde_all, y_tilde_all):
            beta_tilde.append(np.linalg.inv(weight * X_tilde.T @ X_tilde) @ (weight * X_tilde.T @ y_tilde))

        betas_tilde.append(np.array(beta_tilde))

    return {name: np.array(val) for name, val in zip(ys_tilde, betas_tilde)}


def caculate_weights_and_lamdas(
    Xs_tilde: Dict[str, np.ndarray],
    ys_tilde: Dict[str, np.ndarray],
    betas_em: Dict[str, np.ndarray],
    sigmas: Dict[str, np.ndarray],
    alpha: float,
    gamma: float,
    mus_mle: Dict[str, np.ndarray] = None
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Calculates the weights and lambdas using the given data and parameters.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    ys_tilde: Dict[str, np.ndarray]
        Dictionary of ys tilde arrays.
    betas_em: Dict[str, np.ndarray]
        Dictionary of beta_em arrays.
    sigmas: Dict[str, np.ndarray]
        Dictionary of sigma_mle arrays.
    alpha: float
        Alpha parameter value.
    gamma: float
        Gamma parameter value.
    mus_mle: Dict[str, np.ndarray] = None
        Dictionary of mus mle arrays.

    Returns:
    ----------
    Tuple of dictionaries containing the calculated weights and lambdas.

    weights: Dict[str, np.ndarray]
        The weight of each data point.
    lambdas: Dict[str, np.ndarray]
        The weight of each residue
    """

    weights = {name: None for name in ys_tilde}
    lambdas = {name: None for name in ys_tilde}
    
    for name, X_tilde_all, y_tilde_all, beta_em_all, sigma_all in \
        zip(ys_tilde, Xs_tilde.values(), ys_tilde.values(), betas_em.values(), sigmas.values()):
        mu_mle = mus_mle[name] if mus_mle is not None else None
        weight = np.array([np.ones(len(y_tilde_all[0]))] * len(y_tilde_all))
        lambda_ = np.ones(len(beta_em_all))

        for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_em_all, sigma_all)):
            X_tilde, y_tilde, beta_em, sigma_mle = elements

            # Calculate weight
            exponential_w = np.exp(-(alpha / (2 * sigma_mle)) * (y_tilde - X_tilde @ beta_em)**2)
            weight[idx] =  exponential_w / np.sum(exponential_w) * len(y_tilde) 

            # Calculate lambda
            if mu_mle is not None:
                coeff_l = ((1 + gamma) / ((2 * np.pi)**(gamma / 2) * sigma_mle**(gamma / 2 + 1)))
                exponential_l = np.exp(-(gamma / (2 * sigma_mle)) * ((beta_em - mu_mle).T @ np.eye(2) @ (beta_em - mu_mle)))
                lambda_[idx] =  coeff_l * exponential_l
        
        weights[name] = weight
        lambdas[name] = lambda_

    return weights, lambdas


def caculate_points(Xs_tilde: Dict[str, np.ndarray], betas_em: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Calculates the points using the given Xs tilde and beta_em.

    Params
    ----------
    Xs_tilde: Dict[str, np.ndarray]
        Dictionary of Xs tilde arrays.
    betas_em:  Dict[str, np.ndarray]
        Dictionary of beta_em arrays.

    Returns
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated points.

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
    distances_to_center: Dict[str, np.ndarray],
    betas: Dict[str, np.ndarray]
) -> Dict[str, np.ndarray]:
    """
    Calculates the densities using the given distances to center and betas.

    Params
    ----------
    distances_to_center: Dict[str, np.ndarray]
        Dictionary of distances to center arrays.
    betas: Dict[str, np.ndarray]
        Dictionary of beta arrays.

    Returns
    ----------
    Dict[str, np.ndarray]
        Dictionary containing the calculated densities.

    """

    densities = {name: None for name in distances_to_center}

    # Calculate densities for each group
    for name in distances_to_center:
        distance_to_center = np.unique(distances_to_center[name][0])
        X_tilde = np.array([np.ones(len(distance_to_center)), -1 / 2 * distance_to_center**2])
        densities[name] = np.exp(X_tilde.T @ betas[name])

    return densities


def caculate_similarity(
    densities_data: Dict[str, List[List[float]]],
    densities_estimated: Dict[str, List[List[float]]]
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

    for name in densities_data:

        qscores = []

        # Calculate similarity score for each data and estimated density pair
        for data_density, estimated_density in zip(densities_data[name], densities_estimated[name]):
            data_density = np.array(data_density)
            estimated_density = np.array(estimated_density)
            numerator = np.dot(data_density - data_density.mean(), estimated_density - estimated_density.mean())
            denominator = np.sqrt(sum((data_density - data_density.mean())**2)) * np.sqrt(sum((estimated_density - estimated_density.mean())**2))
            qscores.append(numerator / denominator)

        qscores_all[name] = qscores

    return qscores_all