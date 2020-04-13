import numpy as np
from scipy.stats.distributions import gamma, lognorm


def lognorm_pdf(mean, sd):
    """Define a log-normal PDF from its mean and standard deviation"""
    sigma = np.sqrt(np.log(1 + sd**2 / mean**2))
    mu = np.log(mean) - (sigma**2 / 2)
    return lambda x: lognorm.pdf(x, s=sigma, scale=np.exp(mu))


def gamma_pdf(mean, sd):
    """Define a gamma distribution PDF from its mean and standard deviation"""
    coef_of_variation = sd / mean
    alpha = 1 / coef_of_variation**2
    beta = alpha / mean
    return lambda x: gamma.pdf(x, a=alpha, scale=1/beta)
