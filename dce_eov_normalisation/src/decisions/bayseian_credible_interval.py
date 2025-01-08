import numpy as np
from numpy.typing import ArrayLike
import arviz as az
import pandas as pd
import matplotlib.pyplot as plt 


def calculate_posterior(
    data: ArrayLike,
    prior_mean:float,
    prior_var:float,
    data_var:float
    ) -> tuple[float, float]:
    """
    Calculate the posterior mean and variance given prior and data.
    This calculation is based on the conjugate prior theory,
    where the prior normal distribution and known variance.

    Args:
        data: array of data points.
        prior_mean: mean of the prior normal distribution.
        prior_var: variance of the prior normal distribution.
        data_var: variance of the data.
    """
    n = len(data) # type: ignore
    data_mean = np.mean(data)

    # Calculating posterior parameters
    posterior_var = (1 / prior_var + n / data_var)**(-1)
    posterior_mean = posterior_var * (prior_mean / prior_var + n * data_mean / data_var)

    return posterior_mean, posterior_var


def get_credible_interval(
    posterior_mean: float,
    posterior_std: float,
    hdi_prob=0.995
    ) -> tuple[float, float]:
    """
    Calculate the credible interval for the posterior distribution.

    Args:
        posterior_mean: mean of the posterior distribution.
        posterior_std: standard deviation of the posterior distribution.
        hdi_prob: probability of the credible interval.
    """
    posterior_data = np.random.normal(posterior_mean, posterior_std, 1000)
    bounds = az.hdi(posterior_data, hdi_prob=hdi_prob)
    lower_bound, upper_bound = bounds[0], bounds[1]
    return lower_bound, upper_bound

def get_sequential_credible_intervals(
    data: ArrayLike,
    prior_mean: float,
    prior_var: float,
    data_var: float,
    hdi_prob: float = 0.995
    ) -> pd.DataFrame:
    """
    """
    sizes = range(0, len(data) + 1, 12) # type: ignore
    means = []
    posterior_vars = []
    posterior_stds = []
    lower_bounds = []
    upper_bounds = []

    for size in sizes:
        sample_data = data[:size] # type: ignore
        posterior_mean, posterior_var = calculate_posterior(sample_data, prior_mean, prior_var, data_var)
        means.append(posterior_mean)

        # Calculate credible interval
        posterior_std = np.sqrt(posterior_var)
        bounds = get_credible_interval(posterior_mean, posterior_std, hdi_prob=hdi_prob)
        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])
        posterior_vars.append(posterior_var)
        posterior_stds.append(posterior_std)

    # Adjusting error bar lengths
    lower_errors = [mean - lb for mean, lb in zip(means, lower_bounds)]
    upper_errors = [ub - mean for mean, ub in zip(means, upper_bounds)]
    return pd.DataFrame({
        'size': sizes,
        'posterior_mean': means,
        'posterior_var': posterior_vars,
        'posterior_std': posterior_stds,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds,
        'lower_error': lower_errors,
        'upper_error': upper_errors
    })

def plot_sequential_credible_intervals(
    data: ArrayLike,
    prior_mean: float,
    prior_var: float,
    data_var: float,
    hdi_prob: float = 0.995,
    max_days = 31
    ):
    """
    Plot the sequential credible intervals for sequentially adding data to the considered monitoring parameter.

    Args:
        data (ArrayLike): New considered data to be added sequentially.
        prior_mean (float): prior mean of the normal distribution.
        prior_var (float): prior variance of the normal distribution.
        data_var (float): variance of the data.
        hdi_prob (float, optional): Probability for the credible interval.
            Defaults to 0.90.
    """    

    df_sequential_ci = get_sequential_credible_intervals(data, prior_mean, prior_var, data_var, hdi_prob=hdi_prob)
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.grid()
    plt.fill_between(df_sequential_ci['size'], df_sequential_ci['lower_bound'], df_sequential_ci['upper_bound'], color='lightgrey')
    plt.scatter(df_sequential_ci['size'], df_sequential_ci['posterior_mean'], s=30, color='black')
    #plt.errorbar(df_sequential_ci['size'], df_sequential_ci['mean'], yerr=[df_sequential_ci['lower_error'], df_sequential_ci['upper_error']], fmt='o', linewidth=1.0) #type: ignore
    plt.xlabel('Sample Size')
    plt.ylabel('Posterior Mean')
    plt.title(f'{int(hdi_prob * 100)}% Credible Interval for Mean Residuals')
    plt.show()