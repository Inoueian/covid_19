import numpy as np
from scipy.stats.distributions import gamma, lognorm, weibull_min


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


def weibull_pdf(shape, scale):
    """Define a Weibull distribution PDF from its shape and scale parameters"""
    return lambda x: weibull_min.pdf(x, c=shape, scale=scale)


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


def create_infection_tensor(generation_array, num_days):
    """Given num_days,
    return array of infection matrices so that
    multiplying the initial vector [1., 0., 0., ...]
    by result[0] * R0 and adding the last vector,
    multiplying by result[1] * R0 and adding the last vector, ...
    recursively results in the predicted number of new infections for each day."""
    list_of_matrices = []
    for index in range(num_days):
        non_zero_row = np.concatenate([np.zeros(index),
                                       generation_array[:num_days-index]]).reshape((1, num_days))
        matrix = np.vstack([np.zeros((index, num_days)),
                            non_zero_row,
                            np.zeros((num_days - index - 1, num_days))])
        list_of_matrices.append(matrix)
    return np.stack(list_of_matrices)


def predict_N_exponential(N_0, growth_constant, num_days):
    """Given N_0 and growth_constant,
    predict the number of infections up to some point in time, given by num_days,
    using the exponential growth model."""
    return N_0 * np.exp(growth_constant * np.arange(num_days))


def predict_N_generation(N_0, R_0, num_days, generation_pdf):
    """Given N_0 and R_0,
    predict the number of infections up to some point in time, given by num_days,
    using the generation interval model.

    The generation interval distribution is specified by the PDF,
    a function of the number of days since infection."""
    N_array = N_0 * np.array([1.] + [0.] * (num_days - 1))

    generation_array = np.array([generation_pdf(x)
                                 for x in range(num_days)])
    infection_tensor = create_infection_tensor(generation_array, num_days)
    for tensor in infection_tensor[:-1]:
        N_array += R_0 * np.matmul(N_array, tensor)

    return N_array


def predict_N_change_point(N_0, R_0, R_ratio, change_point,
                           num_days, generation_pdf):
    """Given N_0, R_0, R_ratio, and change_point
    predict the number of infections up to some point in time, given by num_days,
    using the change point model.

    The generation interval distribution is specified by the PDF,
    a function of the number of days since infection."""
    R = np.array([R_0 if day < change_point
                  else R_0 * R_ratio
                  for day in range(num_days)])
    R_mat = np.diag(R)

    generation_array = np.array([generation_pdf(x)
                                 for x in range(num_days)])
    infection_tensor = create_infection_tensor(generation_array, num_days)

    N_array = N_0 * np.array([1.] + [0.] * (num_days - 1))
    for tensor in infection_tensor[:-1]:
        N_array += np.matmul(np.matmul(N_array, tensor),
                             R_mat)
    return N_array


def batch_predict_N_exponential(initial_array, growth_array, num_days):
    """Do prediction of the number of infections for a batch of parameters,
    using the exponential model.

    The output has shape (batch_size, num_days)"""
    return np.array([predict_N_exponential(N_0, growth_constant, num_days)
                     for N_0, growth_constant in zip(initial_array,
                                                     growth_array)])


def batch_predict_N_generation(initial_array, R_0_array, num_days,
                               generation_pdf):
    """Do prediction of the number of infections for a batch of parameter,
    using the generation interval model.

    The output has shape (batch_size, num_days)"""
    return np.array([predict_N_generation(N_0, R_0, num_days,
                                          generation_pdf)
                     for N_0, R_0 in zip(initial_array, R_0_array)])


def batch_predict_N_change_point(initial_array, R_0_array,
                                 R_ratio_array, change_point_array, num_days,
                                 generation_pdf):
    """Do prediction of the number of infections for a batch of parameter,
    using the generation interval model.

    The output has shape (batch_size, num_days)"""
    return np.array([predict_N_change_point(N_0, R_0, R_ratio, change_point,
                                            num_days, generation_pdf)
                     for N_0, R_0, R_ratio, change_point in zip(initial_array,
                                                                R_0_array,
                                                                R_ratio_array,
                                                                change_point_array[0])])


def predict_deaths_from_infections(infection_array,
                                   incubation_pdf, delay_pdf,
                                   num_days=None, IFR=0.0095):
    """Given the array of new infections, and parameters that determine
    the propagation of cases to death,
    predict the expected number of deaths up to some point in time, given by num_days
    using the exponential growth model.

    Calculating a whole range of days is more efficient than
    calculating one day at a time.

    num_days can either be specified,
    or left unspecified, when it will be inferred from length of infection_array."""
    if num_days is None:
        num_days = len(infection_array)
    incubation_array = np.array([incubation_pdf(x) for x in range(num_days)])
    incubation_mat = create_delay_matrix(incubation_array)
    delay_array = np.array([delay_pdf(x) for x in range(num_days)])
    delay_mat = create_delay_matrix(delay_array)

    transfer_mat = np.matmul(incubation_mat, delay_mat)

    return IFR * np.matmul(infection_array, transfer_mat)


def predict_deaths_exponential(N_0, growth_constant,
                               incubation_pdf, delay_pdf, num_days,
                               IFR=0.0095):
    """Given N_0, growth_constant, and parameters that determine
    the propagation of cases to death,
    predict the expected number of deaths up to some point in time, given by num_days
    using the exponential growth model.
    """
    infection_array = N_0 * np.exp(growth_constant * np.arange(num_days))
    return predict_deaths_from_infections(infection_array,
                                          incubation_pdf, delay_pdf, num_days,
                                          IFR)


def batch_predict_deaths_from_infections(infection_matrix,
                                         incubation_pdf, delay_pdf,
                                         IFR=0.0095):
    """Given a matrix of infections of shape (batch size, num_days),
    predict the number of deaths for each day.

    Output shape is also (batch size, num_days)"""
    num_days = infection_matrix.shape[1]

    incubation_array = np.array([incubation_pdf(x) for x in range(num_days)])
    incubation_mat = create_delay_matrix(incubation_array)
    delay_array = np.array([delay_pdf(x) for x in range(num_days)])
    delay_mat = create_delay_matrix(delay_array)

    transfer_mat = np.matmul(incubation_mat, delay_mat)

    return IFR * np.matmul(infection_matrix, transfer_mat)


def batch_predict_deaths_exponential(initial_array, growth_array,
                                     incubation_pdf, delay_pdf, num_days,
                                     IFR=0.0095):
    """Do prediction of the number of deaths for a batch of parameters.

    The output has shape (batch size, num_days)"""
    N_matrix = batch_predict_N_exponential(initial_array, growth_array, num_days)

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
