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


def caculcate_bs_tilde_single(
        X_tilde_all,
        y_tilde_all, 
        lambda_all, 
        weight_matrix_all, 
        sigma_matrix, 
        v, 
        s, 
        mu_tilde_all, 
        mu, 
        name
        ):
    
    bs_tilde = []
    b_tilde = []
    for weight_matrix, X_tilde, y_tilde, mu_tilde, lambda_ in zip(weight_matrix_all, X_tilde_all, y_tilde_all, mu_tilde_all, lambda_all):
        b = v * s / 2
        first = mu_tilde.T @ (X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix) @ mu_tilde
        second = y_tilde.T @ weight_matrix @ y_tilde
        last = lambda_ * mu.T @ sigma_matrix @ mu
        b_tilde.append(b + 1 / 2 * (-first + second + last))
    bs_tilde.append(np.array(b_tilde))
    return {name: np.array(bs_tilde)}


def caculate_mus_tilde_single(
        X_tilde_all,
        y_tilde_all, 
        weight_matrix_all, 
        lambda_all, 
        sigma_matrix, 
        mu, 
        name
        ):
    mu_tilde = []
    for weight_matrix, X_tilde, y_tilde, lambda_ in zip(weight_matrix_all, X_tilde_all, y_tilde_all, lambda_all):
        mu_tilde.append(np.linalg.inv(X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix) @ \
                        (X_tilde.T @ weight_matrix @ y_tilde + lambda_ * sigma_matrix @ mu))
    return {name: np.array(mu_tilde)}

def caculate_mu_mle_single(
        X_tilde_all,
        y_tilde_all,
        weight_matrix_all,
        lambda_all,
        sigma_matrix,
        mu,
        v,
        s,
        name, 
):
    mu_tilde_all = caculate_mus_tilde_single(X_tilde_all, y_tilde_all, weight_matrix_all, lambda_all, sigma_matrix, mu, name)[name]
    
    sigma_matrix_tilde_list = []
    a_tilde_list = []
    b_tilde_list = []
    weight_list = []
    inverse_weight_list = []
    for weight_matrix, X_tilde, y_tilde, mu_tilde, lambda_ in zip(weight_matrix_all, X_tilde_all, y_tilde_all, mu_tilde_all, lambda_all):
        
        sigma_matrix_tilde_list.append(lambda_ * (X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix))

        a_tilde = (v + len(X_tilde)) / 2
        a_tilde_list.append(a_tilde)

        b = v * s / 2
        first = mu_tilde.T @ (X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix) @ mu_tilde
        second = y_tilde.T @ weight_matrix @ y_tilde
        last = lambda_ * mu.T @ sigma_matrix @ mu
        b_tilde = b + 1 / 2 * (-first + second + last)
        b_tilde_list.append(b_tilde)

        A = np.linalg.inv(X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix)
        weight = a_tilde / b_tilde * lambda_ * sigma_matrix @ A @ X_tilde.T @ weight_matrix @ y_tilde
        inverse_weight = a_tilde / b_tilde * lambda_ * sigma_matrix @ \
                                                np.linalg.inv(np.eye(2) - lambda_ * A @ sigma_matrix)
        weight_list.append(weight)
        inverse_weight_list.append(inverse_weight)

    mus_mle = {name: np.linalg.inv(sum(inverse_weight_list)) @ sum(weight_list)}
    sigma_matrixs_tilde = {name: np.array(sigma_matrix_tilde_list)}
    as_tilde = {name: np.array(a_tilde_list)}
    bs_tilde = {name: np.array(b_tilde_list)}

    return mu_tilde_all, mus_mle, sigma_matrixs_tilde, as_tilde, bs_tilde


def caculate_mu_mle(
        Xs_tilde,
        ys_tilde,
        weight_matrixs,
        lambdas,
        sigma_matrixs,
        mus,
        vs,
        ss,
):
    mus_tilde_dict = {}
    mus_mle_dict = {}
    sigma_matrixs_tilde_dict = {}
    as_tilde_dict = {}
    bs_tilde_dict = {}

    for y_tilde_all, lambda_all, weight_matrix_all, sigma_matrix, v, s, mu, X_tilde_all, name in \
        zip(ys_tilde.values(), lambdas.values(), weight_matrixs.values(), sigma_matrixs.values(), \
            vs.values(), ss.values(), mus.values(), Xs_tilde.values(), Xs_tilde.keys()):

        mus_tilde_all, mus_mle_all, sigma_matrixs_tilde_all, as_tilde_all, bs_tilde_all = caculate_mu_mle_single(
            X_tilde_all, y_tilde_all, weight_matrix_all, lambda_all, sigma_matrix, mu, v, s, name
            )
        
        mus_tilde_dict.update(mus_tilde_all)
        mus_mle_dict.update(mus_mle_all)
        sigma_matrixs_tilde_dict.update(sigma_matrixs_tilde_all)
        as_tilde_dict.update(as_tilde_all)
        bs_tilde_dict.update(bs_tilde_all)

    return mus_tilde_dict, mus_mle_dict, sigma_matrixs_tilde_dict, as_tilde_dict, bs_tilde_dict

def negative_log_likelihood(
        v, 
        s, 
        sigma_matrix,
        Xs_tilde, 
        ys_tilde, 
        lambdas, 
        weight_matrixs, 
        mus,
        name
):
    ans = 0
    X_tilde_all, y_tilde_all, lambda_all, weight_matrix_all, mu =\
    Xs_tilde[name], ys_tilde[name], lambdas[name], weight_matrixs[name], mus[name]
    mu_tilde_all = caculate_mus_tilde_single(X_tilde_all, y_tilde_all, weight_matrix_all, lambda_all, sigma_matrix, mu, name)[name]
    b_tilde_all = caculcate_bs_tilde_single(X_tilde_all, y_tilde_all, lambda_all, weight_matrix_all, sigma_matrix, v, s, mu_tilde_all, mu, name)[name]
    for X_tilde, b_tilde, weight_matrix, lambda_ in zip(X_tilde_all, * b_tilde_all, weight_matrix_all, lambda_all):
        first = np.log(np.linalg.det(X_tilde.T @ weight_matrix @ X_tilde + lambda_ * sigma_matrix))
        # second = np.log(np.linalg.det(weight_matrix))
        a = v / 2
        b = v * s / 2
        a_tilde = (v + len(X_tilde)) / 2

        third = a * np.log(b)
        forth = loggamma(a)
        fifth = loggamma(a_tilde)
        sixth = a_tilde * np.log(b_tilde)

        ans += 1 / 2 * first + third - forth + fifth - sixth
    return -ans

def estimator_negative_log_likelihood( 
        params, 
        Xs_tilde, 
        ys_tilde, 
        lambdas, 
        weight_matrixs, 
        mus,
        name
        ):
    
    # sigma_matrix = np.diag(params[: 2])
    sigma_matrix = np.eye(2)
    v = params[0]
    s = params[1]
    return negative_log_likelihood(v, s, sigma_matrix, Xs_tilde, ys_tilde, \
                                   lambdas, weight_matrixs, mus, name)

def empirical_bayes_single(mu_tilde_all, as_tilde_all, bs_tilde_all):
    sigma_empirical_bayes = []
    for a_tilde, b_tilde in zip(*as_tilde_all.values(), *bs_tilde_all.values()):
        sigma_empirical_bayes.append(b_tilde / (a_tilde - 1))
    return mu_tilde_all, sigma_empirical_bayes


def caculate_weight_matrix(X_tilde_all, y_tilde_all, betas_em, sigmas_em, alpha):
    weights = [0] * len(betas_em)
    for idx, elements in enumerate(zip(X_tilde_all, y_tilde_all, betas_em, sigmas_em)):
        X_tilde, y_tilde, beta_em, sigma_em = elements
        coeff = ((1 + alpha) / ((2 * np.pi)**(alpha / 2) * sigma_em**(alpha / 2 + 1)))
        exponential = np.exp(-(alpha / (2 * sigma_em)) * (y_tilde - X_tilde @ beta_em)**2)
        # print((y_tilde - X_tilde @ beta_em)**2)
        # print(sigma_em)
        weights[idx] = np.diag(coeff * exponential)
    return weights


def caculate_lambdas(mu_mle, sigma_matrix, betas_em, sigmas_em, alpha):
    lambda_ = np.ones(len(betas_em))
    for idx, elements in enumerate(zip(betas_em, sigmas_em)):
        beta_em, sigma_em = elements
        coeff = ((1 + alpha) / ((2 * np.pi)**(alpha / 2) * sigma_em**(alpha / 2 + 1)))
        exponential = np.exp(-(alpha / (2 * sigma_em)) * ((beta_em - mu_mle).T @ sigma_matrix @ (beta_em - mu_mle)))
        # print(beta_em - mu_mle)
        # print(sigma_matrix)
        # print(sigma_em)
        lambda_[idx] = coeff * exponential
    # lambda_ = lambda_ / np.sum(lambda_)
    return lambda_