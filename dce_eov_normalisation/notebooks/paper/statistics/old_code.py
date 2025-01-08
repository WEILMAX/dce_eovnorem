import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_posterior(data, prior_mean, prior_var, data_var):
    """
    Calculate the posterior mean and variance given prior and data.
    """
    n = len(data)
    data_mean = np.mean(data)
    data_var = prior_var

    # Calculating posterior parameters
    posterior_var = (1 / prior_var + n / data_var)**(-1)
    posterior_mean = posterior_var * (prior_mean / prior_var + n * data_mean / data_var)

    return posterior_mean, posterior_var

def get_credible_interval(data, prior_mean, prior_var, data_var, confidence_level=0.90):
    """
    Calculate the credible interval for the posterior distribution.
    """
    post_mean, post_var = calculate_posterior(data, prior_mean, prior_var, data_var)
    lower_bound, upper_bound = stats.norm.interval(confidence_level, loc=post_mean, scale=np.sqrt(post_var))
    return lower_bound, upper_bound

def plot_credible_interval(data, prior_mean, prior_var, data_var, confidence_level=0.90):
    """
    Plot the credible interval for different sizes of aggregated data.
    """
    sizes = range(10, len(data) + 1, 10)
    means = []
    lower_bounds = []
    upper_bounds = []

    for size in sizes:
        sample_data = data[:size]
        post_mean, post_var = calculate_posterior(sample_data, prior_mean, prior_var, data_var)
        means.append(post_mean)

        # Calculate credible interval
        lower_bound, upper_bound = stats.norm.interval(confidence_level, loc=post_mean, scale=np.sqrt(post_var))
        lower_bounds.append(lower_bound)
        upper_bounds.append(upper_bound)

    # Adjusting error bar lengths
    lower_errors = [mean - lb for mean, lb in zip(means, lower_bounds)]
    upper_errors = [ub - mean for mean, ub in zip(means, upper_bounds)]

    # Plotting
    plt.figure(figsize=(20, 6))
    plt.grid()
    plt.errorbar(sizes, means, yerr=[lower_errors, upper_errors], fmt='o', linewidth=0.2)
    plt.xlabel('Sample Size (days)')
    plt.ylabel('Posterior Mean')
    plt.title(f'{int(confidence_level * 100)}% Credible Interval for Mean Residuals')
    # transform xticks in days, knowing that sample frequency is 10min, so 144 samples per day
    plt.xticks(np.arange(0, len(data), 144), np.arange(0, len(data)//144 + 1, 1))
    plt.show()


import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

def calculate_posterior(data, prior_mean, prior_var, data_var):
    """
    Calculate the posterior mean and variance given prior and data.
    """
    n = len(data)
    data_mean = np.mean(data)
    data_var = prior_var

    # Calculating posterior parameters
    posterior_var = (1 / prior_var + n / data_var)**(-1)
    posterior_mean = posterior_var * (prior_mean / prior_var + n * data_mean / data_var)

    return posterior_mean, posterior_var

def get_credible_interval(data, prior_mean, prior_var, data_var, confidence_level=0.90):
    """
    Calculate the credible interval for the posterior distribution.
    """
    post_mean, post_var = calculate_posterior(data, prior_mean, prior_var, data_var)
    lower_bound, upper_bound = stats.norm.interval(confidence_level, loc=post_mean, scale=np.sqrt(post_var))
    return lower_bound, upper_bound

def plot_credible_interval(data, prior_mean, prior_var, data_var, confidence_level=0.90):
    """
    Plot the credible interval for different sizes of aggregated data.
    """
    import arviz as az

    sizes = range(10, len(data) + 1, 10)
    means = []
    lower_bounds = []
    upper_bounds = []

    for size in sizes:
        sample_data = data[:size]
        post_mean, post_var = calculate_posterior(sample_data, prior_mean, prior_var, data_var)
        means.append(post_mean)

        # Calculate credible interval
        posterior_data = np.random.normal(post_mean, np.sqrt(post_var), 1000)
        bounds = az.hdi(posterior_data, hdi_prob=.95)
        lower_bounds.append(bounds[0])
        upper_bounds.append(bounds[1])

    # Adjusting error bar lengths
    lower_errors = [mean - lb for mean, lb in zip(means, lower_bounds)]
    upper_errors = [ub - mean for mean, ub in zip(means, upper_bounds)]
    print(lower_errors, upper_errors)
    plt.plot(lower_errors, 'o')
    plt.plot(upper_errors, 'o')
    plt.show()
    # Plotting
    plt.figure(figsize=(20, 6))
    plt.grid()
    plt.errorbar(sizes, means, yerr=[lower_errors, upper_errors], fmt='o', linewidth=0.5)
    plt.xlabel('Sample Size (days)')
    plt.ylabel('Posterior Mean')
    plt.title(f'{int(confidence_level * 100)}% Credible Interval for Mean Residuals')
    # transform xticks in days, knowing that sample frequency is 10min, so 144 samples per day
    plt.xticks(np.arange(0, len(data), 144), np.arange(0, len(data)//144 + 1, 1))
    plt.show()
