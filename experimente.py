"""
Generate and fit a toy pseudoexperiment.
"""
import math
import random
import numpy
from numpy.random import default_rng
import scipy

from iminuit import Minuit

import matplotlib.pyplot as plt

random.seed(137)  # for reproducibility
numpy_rng = default_rng(12345)  # for reproducibility

# Module data: these are values that were pre-determined for this use.
# See the README.md file for some explanation

# The values used for generating pseudoexperiments

A = 10.26
dA = 0.3
B = 5.16
dB = 0.1
C = 3.31
dC = 0.6
D = 0.76
dD = 0.04

# The number of bins in our energy spectra
N_b = 20

# The experiment's observed energy spectrum
OBS = [7, 4, 4, 3, 4, 6, 5, 3, 6, 5, 4, 1, 3, 0, 1, 1, 2, 0, 1, 0]


class LogLike:
    def __init__(self, data, mass=None, delta=None):
        self.data = data
        self.mass = mass
        self.delta = delta

    def __call__(self, a, b, c, d, mass=None, delta=None):
        mass = self.mass if mass is None else mass
        delta = self.delta if delta is None else delta
        return self.negative_log_likelihood(a, b, c, d, mass, delta)

    # ...

    def negative_log_likelihood(self, a, b, c, d, mass, delta):
        bin_poisson_means = [
            poisson_mean_model(a, b, c, d, mass, delta, k)
            for k in range(1, N_b + 1)
        ]

        #bin_log_likelihoods = [
        #    mu - d * math.log(mu) + math.log(numpy.math.factorial(d))
        #    for mu, d in zip(bin_poisson_means, self.data)
        #]
        small_constant = 1e-10 # you may need to adjust this value

        bin_log_likelihoods = [
            mu - d * math.log(mu + small_constant) if mu > 0 else 0
            for mu, d in zip(bin_poisson_means, self.data)
        ]
        #bin_log_likelihoods = [
        #    mu - d * math.log(mu) if mu > 0 else 0#abs(mu) - d * math.log(abs(mu))
        #    for mu, d in zip(bin_poisson_means, self.data)
        #]
        return sum(bin_log_likelihoods)

    def restricted_numpy(self, x):
        """Return the negative log likelihood for the encapsulated data, but
        with the mass and delta parameters fixed to specific values.
        """
        return self.restricted(*list(x))

    def restricted(self, a, b, c, d):
        """Return the negative log likelihood for the encapsulated data, but
        with the mass and delta parameters fixed to specified values.
        """
        bin_poisson_means = [
            poisson_mean_model(a, b, c, d, self.mass, self.delta, k)
            for k in range(1, N_b + 1)
        ]

        #bin_log_likelihoods = [
        #    mu - d * math.log(mu) + math.log(numpy.math.factorial(d))
        #    for mu, d in zip(bin_poisson_means, self.data)
        #]
        #bin_log_likelihoods = [
        #    mu - d * math.log(mu) if mu > 0 else 0#abs(mu) - d * math.log(abs(mu))
        #    for mu, d in zip(bin_poisson_means, self.data)
        #]

        small_constant = 1e-10
        bin_log_likelihoods = []
        for mu, d in zip(bin_poisson_means, self.data):
            if mu == 'inf' or mu == 'nan' or mu < 0:
                #print(f"mu: {mu}, d: {d}")
                continue
            value = mu - d * math.log(mu + small_constant) if mu > 0 else 0
            bin_log_likelihoods.append(value)
        return sum(bin_log_likelihoods)

def poisson_mean_model(a, b, c, d, mass, delta, k):
    """Generate the Poisson mean for bin k, for the given set of parameter
    values.
    """
    try:
        background = a * math.exp(-k / b) + d
    except OverflowError:
        #print(f"OverflowError with a={a}, b={b}, c={c}, d={d}, mass={mass}, delta={delta}, k={k}")
        background = float('inf')  # or some large number
    signal = (c / delta) * math.exp(-0.5 * ((k - mass) / delta) ** 2)
    return background + signal

def generate_pseudoexperiment(m, Delta):
    """Generate a random array of bin counts (a pseudoexperiment) for one
    pseudoexperiment, at the location (m, Delta) in our parameter space.
    """
    current_A = random.normalvariate(A, dA)
    current_B = random.normalvariate(B, dB)
    current_C = random.normalvariate(C, dC)
    current_D = random.normalvariate(D, dD)
    current_means = [
        poisson_mean_model(
            current_A, current_B, current_C, current_D, m, Delta, k
        )
        for k in range(1, N_b + 1)
    ]
    # We are not using numpy's ability to generate an array of variates
    # in a single call because we are trying to have a "pure python"
    # solution. However, since Python does not have a built-in Poisson
    # random number generate, we use one from numpy.
    # N.B.: this is not how we would use this generator if we were
    # trying to write efficient code.
    #generated_counts = [numpy_rng.poisson(bin_mean) for bin_mean in current_means]
    generated_counts = [numpy_rng.poisson(bin_mean) for bin_mean in current_means if bin_mean > 0]
    return generated_counts


def fit_given_data(data):
    """Given data, find the minimum of the negative log likelihood."""
    loglike = LogLike(data)
    initial_guess = dict(a=10.0, b=5.0, c=3.0, d=1.0, mass=8.0, delta=2.0)
    m = Minuit(loglike, **initial_guess) # Minuit class
    #m = Minuit(loglike, **initial_guess, 
    #        limit_a=(0, None), limit_b=(1e-6, None), limit_c=(0, None), limit_d=(0, None), 
    #        #limit_mass=(mass_min, mass_max), limit_delta=(delta_min, delta_max),
    #        errordef=Minuit.LIKELIHOOD)


    # Use the Migrad minimization algorithm, exclusive usage in high-energy physics
    m.migrad()

    # store the results in a dictionary
    fit_result = {}
    fit_result['values'] = m.values
    fit_result['errors'] = m.errors
    fit_result['success'] = m.fmin.is_valid
    fit_result['fun'] = m.fmin.fval

    return fit_result

def fit_given_data_at_location(mass, delta, data):
    """Given values for m and Delta, fit the remaining parameters to the given data."""
    loglike = LogLike(data, mass, delta)
    initial_guess = dict(a=10.0, b=5.0, c=3.0, d=1.0)
    m = Minuit(loglike.restricted, **initial_guess)
    #m = Minuit(loglike, **initial_guess, 
    #        limit_a=(0, None), limit_b=(1e-6, None), limit_c=(0, None), limit_d=(0, None), 
    #        #limit_mass=(mass_min, mass_max), limit_delta=(delta_min, delta_max),
    #        errordef=Minuit.LIKELIHOOD)


    # Use the Migrad minimization algorithm, exclusive usage in high-energy physics
    m.migrad() # Iteratively finds the minimum of a function

    # store the results in a dictionary
    fit_result = {}
    fit_result['values'] = m.values
    fit_result['errors'] = m.errors
    fit_result['success'] = m.fmin.is_valid
    fit_result['fun'] = m.fmin.fval

    return fit_result


def generate_one_fit(m, Delta):
    """Generate a single fit to the log likelihood, by generating and fitting
    one pseudoexperiment.
    """
    pe = generate_pseudoexperiment(m, Delta)
    fit = fit_given_data(pe)
    return fit


if __name__ == "__main__":
    # Find the best fit (allowing all of the parameters, both physics and
    # nuisance, to vary).
    best_fit = fit_given_data(OBS)
    assert best_fit['success']
    lambda_best = best_fit['fun']
    params_best = best_fit['values']

    # Pick a spot in our parameter space. At this location, we are going to
    # go through the profiled FC procedure. The result will be a p-value
    # (probabililty) at this location in the parameter space.
    #m_p, Delta_p = 8.0, 2.0

    #masses = [8.0]#,7.0,6.0]
    #deltas = [2.0]#,3.0,4.0]

    num_of_pseudo = 5

    # Define mass and delta ranges
    mass_values = numpy.linspace(5.0, 10.0, num=5)
    delta_values = numpy.linspace(1.0, 4.0, num=5)
    # 5 x 5 x 5 experiments

    likelihood_ratios = {}

    # Iterate over mass and delta ranges
    for mass in mass_values:
        for delta in delta_values:
            # Compute likelihood ratios for each (mass, delta) pair
            likelihood_ratios[(mass, delta)] = []
            for _ in range(num_of_pseudo):
                try:
                    pe = generate_pseudoexperiment(mass, delta)
                    full_fit = fit_given_data(pe)
                    p_fit = fit_given_data_at_location(mass, delta, pe)
                    likelihood_ratio = 2 * (full_fit['fun'] - p_fit['fun'])
                    likelihood_ratios[(mass, delta)].append(likelihood_ratio)
                except OverflowError as e:
                    print(f"OverflowError at mass={mass}, delta={delta}: {e}")

    # Initialize array for contour plot data
    contour_data = numpy.empty((len(mass_values), len(delta_values)))

    for i, mass in enumerate(mass_values):
        for j, delta in enumerate(delta_values):
            critical_value = numpy.percentile(likelihood_ratios[(mass, delta)], 90)  # Compute critical value
            # Compare likelihood ratio of best fit with critical value and store result
            contour_data[i, j] = 1 if lambda_best <= critical_value else 0

    # Generate the contour plot
    X, Y = numpy.meshgrid(mass_values, delta_values)
    #Z = numpy.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    plt.contourf(X, Y, contour_data.T, levels=[0, 0.5, 1], colors=['white', 'blue'])
    #plt.contourf(X, Y, levels=numpy.linspace(start, stop, num_steps), cmap=plt.cm.jet)

    plt.xlabel('Mass')
    plt.ylabel('Delta')
    plt.title('Confidence region')
    plt.show()
