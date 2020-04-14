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


def create_delay_matrix(array, num_days):
    """Given a PDF of the incubation period distribution,
    and the number of days that we want to run the simulation for,
    create a matrix that converts the number of new infections to
    the number of new patients.

    The same logic can be used to turn the number of new patients
    to the number of new deaths (up to the factor of infection fatality rate),
    using the PDF for the delay time between illness onset to death."""
    list_of_lists = [[array[time_1 - time_0] if time_1 > time_0 else 0.
                      for time_1 in range(num_days)]
                     for time_0 in range(num_days)]
    return np.array(list_of_lists)
