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


def create_delay_matrix(array, num_days=None):
    """Given a PDF of the incubation period distribution,
    and the number of days that we want to run the simulation for,
    create a matrix that converts the number of new infections to
    the number of new patients.

    The same logic can be used to turn the number of new patients
    to the number of new deaths (up to the factor of infection fatality rate),
    using the PDF for the delay time between illness onset to death."""
    if num_days is None:
        num_days = len(array)
    list_of_lists = [[array[time_1 - time_0] if time_1 > time_0 else 0.
                      for time_1 in range(num_days)]
                     for time_0 in range(num_days)]
    return np.array(list_of_lists)


def predict_N_exponential(num_days, N_0, growth_constant):
    """Given N_0 and growth_constant,
    predict the number of infections up to some point in time, given by num_days,
    using the exponential growth model"""
    return N_0 * np.exp(growth_constant * np.arange(num_days))


def batch_predict_N_exponential(num_days, initial_array, growth_array):
    """Do prediction of the number of infections for a batch of parameters.

    The output has shape (batch_size, num_days)"""
    return np.array([predict_N_exponential(num_days, N_0, growth_constant)
                     for N_0, growth_constant in zip(initial_array,
                                                     growth_array)])


def predict_deaths_exponential(num_days, N_0, growth_constant,
                               incubation_pdf, delay_pdf,
                               IFR=0.0095):
    """Given N_0, growth_constant, and parameters that determine
    the propagation of cases to death,
    predict the expected number of deaths up to some point in time, given by num_days
    using the exponential growth model.

    Calculating a whole range of days is more efficient than
    calculating one day at a time.
    """
    N_array = N_0 * np.exp(growth_constant * np.arange(num_days))

    incubation_array = np.array([incubation_pdf(x) for x in range(num_days)])
    incubation_mat = create_delay_matrix(incubation_array)
    delay_array = np.array([delay_pdf(x) for x in range(num_days)])
    delay_mat = create_delay_matrix(delay_array)

    transfer_mat = np.matmul(incubation_mat, delay_mat)

    return IFR * np.matmul(N_array, transfer_mat)


def batch_predict_deaths_exponential(num_days, initial_array, growth_array,
                                     incubation_pdf, delay_pdf, IFR=0.0095):
    """Do prediction of the number of infections for a batch of parameters.

    The output has shape (batch_size, num_days)"""
    N_matrix = batch_predict_N_exponential(num_days, initial_array, growth_array)

    incubation_array = np.array([incubation_pdf(x) for x in range(num_days)])
    incubation_mat = create_delay_matrix(incubation_array)
    delay_array = np.array([delay_pdf(x) for x in range(num_days)])
    delay_mat = create_delay_matrix(delay_array)

    transfer_mat = np.matmul(incubation_mat, delay_mat)

    return IFR * np.matmul(N_matrix, transfer_mat)


def mean_prediction(pred_matrix):
    """Given a set of predictions with shape (batch_size, num_days),
    calculate the mean prediction for each day."""
    return np.mean(pred_matrix, axis=0)


def percentile_prediction(pred_matrix, percentile):
    """Given a set of predictions with shape (batch_size, num_days),
    calculate the prediction that is at a given percentile for each day."""
    return np.percentile(pred_matrix, q=percentile, axis=0)
