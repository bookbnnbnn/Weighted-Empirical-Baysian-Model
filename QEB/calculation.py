import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, invgamma
from scipy.special import gamma
from scipy.special import loggamma
from scipy.optimize import minimize
import random
import matplotlib.pyplot as plt
import logging
from collections import Counter
import time 
from numba import njit

def caculate_mus_tilde(
        Xs_tilde,
        ys_tilde, 
        weight_matrixs, 
        lambdas, 
        mus, 
        ):
    
    mus_tilde = []
    for X_tilde_all, y_tilde_all, mu, weight_matrix_all, lambda_all in \
        zip(Xs_tilde.values(), ys_tilde.values(), mus.values(), weight_matrixs.values(), lambdas.values()):
        mu_tilde = []
        for weight_matrix, X_tilde, y_tilde, lambda_ in zip(weight_matrix_all, X_tilde_all, y_tilde_all, lambda_all):
            mu_tilde.append(np.linalg.inv(X_tilde.T @ weight_matrix @ X_tilde + lambda_ * np.eye(2)) @ \
                            (X_tilde.T @ weight_matrix @ y_tilde + lambda_ * np.eye(2) @ mu))
        mus_tilde.append(np.array(mu_tilde))
    return {name: np.array(val) for name, val in zip(ys_tilde, mus_tilde)}


def caluculate_sigmas(Xs_tilde, ys_tilde, betas, initial=False):
    sigmas_initial = []
    for X_tilde_all, y_tilde_all, betas in zip(Xs_tilde.values(), ys_tilde.values(), betas.values()):
        sigma_initial = []
        if initial:
            betas = [betas] * len(y_tilde_all)
        for X_tilde, y_tilde, beta in zip(X_tilde_all, y_tilde_all, betas):
            residual = (y_tilde - X_tilde @ beta)
            sigma_initial.append(np.median(residual ** 2))
        sigmas_initial.append(sigma_initial)
    return {name: np.array(val) for name, val in zip(ys_tilde, sigmas_initial)}


def caculate_mus_mle(Xs_tilde, ys_tilde, sigmas, weight_matrixs, lambdas):
    mus_mle = {name: None for name in ys_tilde}
    for name, X_tilde_all, y_tilde_all, sigma_all, weight_matrix_all, lambda_all in \
        zip(ys_tilde, Xs_tilde.values(), ys_tilde.values(), sigmas.values(), weight_matrixs.values(), lambdas.values()):
        denominators = np.array([np.eye(2)] * len(y_tilde_all))
        numerators = np.array([np.ones(2)] * len(y_tilde_all))
        for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, sigma_all, weight_matrix_all, lambda_all)):          
            X_tilde, y_tilde, sigma, weight_matrix, lambda_ = elements
            A = np.linalg.inv(X_tilde.T @ weight_matrix @ X_tilde + lambda_ * np.eye(2))
            denominator = lambda_ / sigma * (np.eye(2) - A * lambda_)
            numerator = lambda_ / sigma * A @ X_tilde.T @ weight_matrix @ y_tilde
            denominators[idx] = denominator
            numerators[idx] = numerator
        mus_mle[name] = np.linalg.inv(np.sum(denominators, axis=0)) @ np.sum(numerators, axis=0)
    return mus_mle

def caculate_weighted_beta(
        Xs_tilde,
        ys_tilde, 
        weight_matrixs,
        ):
    
    betas_tilde = []
    for X_tilde_all, y_tilde_all, weight_matrix_all in zip(Xs_tilde.values(), ys_tilde.values(), weight_matrixs.values()):
        beta_tilde = []
        for weight_matrix, X_tilde, y_tilde in zip(weight_matrix_all, X_tilde_all, y_tilde_all):
            beta_tilde.append(np.linalg.inv(X_tilde.T @ weight_matrix @ X_tilde) @ (X_tilde.T @ weight_matrix @ y_tilde))
        betas_tilde.append(np.array(beta_tilde))
    return {name: np.array(val) for name, val in zip(ys_tilde, betas_tilde)}


def caculate_weights_and_lamdas(Xs_tilde, ys_tilde, betas_em, sigmas_mle, alpha, gamma, mus_mle=None):
    weights = {name: None for name in ys_tilde}
    lambdas = {name: None for name in ys_tilde}
    for name, X_tilde_all, y_tilde_all, beta_em_all, sigma_mle_all in \
        zip(ys_tilde, Xs_tilde.values(), ys_tilde.values(), betas_em.values(), sigmas_mle.values()):
        mu_mle = mus_mle[name] if mus_mle is not None else None
        weight = np.array([np.eye(len(y_tilde_all[0]))] * len(y_tilde_all))
        lambda_ = np.ones(len(beta_em_all))
        for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, beta_em_all, sigma_mle_all)):
            X_tilde, y_tilde, beta_em, sigma_mle = elements

            coeff_w = ((1 + alpha) / ((2 * np.pi)**(alpha / 2) * sigma_mle**(alpha / 2 + 1)))
            exponential_w = np.exp(-(alpha / (2 * sigma_mle)) * (y_tilde - X_tilde @ beta_em)**2)
            weight[idx] = np.diag(coeff_w * exponential_w) / np.sum(np.diag(coeff_w * exponential_w)) * len(y_tilde) 
            if mu_mle is not None:
                coeff_l = ((1 + gamma) / ((2 * np.pi)**(gamma / 2) * sigma_mle**(gamma / 2 + 1)))
                exponential_l = np.exp(-(gamma / (2 * sigma_mle)) * ((beta_em - mu_mle).T @ np.eye(2) @ (beta_em - mu_mle)))
                lambda_[idx] =  coeff_l * exponential_l
        
        weights[name] = weight
        lambdas[name] = lambda_

    return weights, lambdas

def caculate_points(Xs_tilde, betas_em):
    points = {name: None for name in Xs_tilde}
    for name, X_tilde_all, beta_em_all in zip(Xs_tilde, Xs_tilde.values(), betas_em.values()):
        density = np.array([np.ones(len(X_tilde_all[0]))] * len(X_tilde_all))
        for idx, elements in enumerate(zip(X_tilde_all, beta_em_all)):
            X_tilde, beta_em = elements
            density[idx] = np.exp(X_tilde @ beta_em)
        points[name] = density
            
    return points

def caculate_density(distances_to_center, betas):
    densities = {name: None for name in distances_to_center}
    for name in distances_to_center:
        distance_to_center = np.unique(distances_to_center[name][0])
        X_tilde = np.array([np.ones(len(distance_to_center)), -1 / 2 * distance_to_center**2])
        densities[name] = np.exp(X_tilde.T @ betas[name])
    return densities

def caculate_similarity(densities_data, densities_estimated):
    qscores_all = {}
    for name in densities_data:
        qscores = []
        for data_density, estimated_density in zip(densities_data[name], densities_estimated[name]):
            data_density = np.array(data_density)
            estimated_density = np.array(estimated_density)
            numerator = np.dot(data_density - data_density.mean(), estimated_density - estimated_density.mean())
            denominator = np.sqrt(sum((data_density - data_density.mean())**2)) * np.sqrt(sum((estimated_density - estimated_density.mean())**2))
            qscores.append(numerator / denominator)
        qscores_all[name] = qscores
    return qscores_all