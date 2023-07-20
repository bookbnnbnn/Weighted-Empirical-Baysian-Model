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
        denominator = lambda_ / sigma * (np.eye(2) - sigma_tilde_matrix * lambda_)
        numerator = lambda_ / sigma * sigma_tilde_matrix @ XWY
        denominators[idx] = denominator
        numerators[idx] = numerator

    return np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)


def caculate_betas_WLR(
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
    alpha: float,
    gamma: float,
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
        X_tilde, y_tilde, beta_em, sigma = elements

        # Calculate weight
        exponential_w = np.exp(-(alpha / (2 * sigma)) * (y_tilde - X_tilde @ beta_em)**2)
        weight[idx] =  exponential_w / np.sum(exponential_w) * len(y_tilde) 

        # Calculate lambda
        if mu_mle is not None:
            coeff_l = sigma**(-(gamma + 2) / 2)
            exponential_l = np.exp(-(gamma / (2 * sigma)) * ((beta_em - mu_mle).T @ np.eye(2) @ (beta_em - mu_mle)))
            lambda_[idx] =  coeff_l * exponential_l

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
    betas: np.ndarray
) -> np.ndarray:
    """
    Calculates the densities using the given distances to center and betas.

    Params
    ----------
    distances_to_center: np.ndarray
        distances to center arrays.
    betas: np.ndarray
        beta arrays.

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