"""
Generate and fit a toy pseudoexperiment.
"""
import math
import random
import numpy
from numpy.random import default_rng
import scipy



random.seed(137)  # for reproducibility
np_rng = default_rng(12345)  # for reproducibility

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
    """LogLike is an encapsulation of the (negative) log likelihood function.
    It contains as state the "data" for which this is the likelihood."""

    def __init__(self, data, mass=None, delta=None):
        self.data = data
        self.mass = mass
        self.delta = delta

    def __call__(self, x):
        """Return the negative log likelihood for the encapsulated data,
        for the given set of parameter values in the numpy array `x`.
        N.B.: This is grossly inefficient, and is only being done
        here to keep most of the code in 'normal' Python, rather than using
        numpy.
        """
        return self.negative_log_likelihood(*list(x))

    def restricted_np(self, x):
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

        epsilon = 1e-10  # small constant to avoid log(0) issue
        bin_log_likelihoods = [
            max(mu, epsilon) - d * math.log(max(mu, epsilon)) + math.log(numpy.math.factorial(max(0, d)))
            for mu, d in zip(bin_poisson_means, self.data)
        ]
        return sum(bin_log_likelihoods)

    def negative_log_likelihood(self, a, b, c, d, mass, delta):
        """Return the value of lambda for given set of model parameters,
        and for the fixed data in OBS.
        """
        bin_poisson_means = [
            poisson_mean_model(a, b, c, d, mass, delta, k)
            for k in range(1, N_b + 1)
        ]

        bin_log_likelihoods = [
            mu - d * math.log(mu) + math.log(numpy.math.factorial(d))
            for mu, d in zip(bin_poisson_means, self.data)
        ]
        return sum(bin_log_likelihoods)


def poisson_mean_model(a, b, c, d, mass, delta, k):
    """Generate the Poisson mean for bin k, for the given set of parameter
    values.
    """
    background = a * math.exp(-k / b) + d
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
    #generated_counts = [np_rng.poisson(bin_mean) for bin_mean in current_means]
    generated_counts = [np_rng.poisson(bin_mean) for bin_mean in current_means if bin_mean > 0]
    return generated_counts


def fit_given_data(data):
    """Given data (a list of counts in energy bins, either from a
    pseudoexperiment or real observations), find the
    minimum of the negative log likelihood and return it.
    """
    loglike = LogLike(data)
    initial_guess = numpy.array([10.0, 5.0, 3.0, 1.0, 8.0, 2.0])
    fit_result = scipy.optimize.minimize(loglike, initial_guess)
    return fit_result


def fit_given_data_at_location(mass, delta, data):
    """Given values for m and Delta, fit the remaining parameters to the given
    data.
    """
    loglike = LogLike(data, mass, delta)
    initial_guess = numpy.array([10.0, 5.0, 3.0, 1.0])
    fit_result = scipy.optimize.minimize(loglike.restricted_np, initial_guess)
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
    assert best_fit.success
    lambda_best = best_fit.fun
    params_best = best_fit.x

    # Pick a spot in our parameter space. At this location, we are going to
    # go through the profiled FC procedure. The result will be a p-value
    # (probabililty) at this location in the parameter space.
    m_p, Delta_p = 8.0, 2.0

    masses = [8.0,7.0,6.0]
    deltas = [2.0,3.0,4.0]

    num_of_pseudo = 10
    # 3 x 3 x 10 optimizations

    print(f"Loop starts\n\n")
    for mass in masses:
        for delta in deltas:
            # Now we fix the values of m and Delta, and re-fit the *other* parameters.
            for _ in range(num_of_pseudo):
                # Generate pseudoexperiment
                pe = generate_pseudoexperiment(mass,delta)
                print(f"Generated pseudoexperiment: \n {pe}\n")
                # Fit the data allowing all parameters to vary
                #full_fit = fit_given_data(pe)
                
                p_fit = fit_given_data_at_location(mass, delta, pe)

                #likelihood_ratio = 2 * (full_fit.fun - p_fit.fin)
                print(f"Fit for mass: {mass} and delta: {delta}: \n {p_fit}\n\n")

            # Compute the critical value for 90% confidence level
            #critical_value = numpy.percentile(likelihood_ratios, 90)

