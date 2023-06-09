"""
Generate and fit a toy pseudoexperiment.
"""

import math
import random
from numpy.random import default_rng

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
        poisson_mean_model(current_A, current_B, current_C, current_D, m, Delta, k)
        for k in range(1, N_b + 1)
    ]
    # We are not using numpy's ability to generate an array of variates
    # in a single call because we are trying to have a "pure python"
    # solution. However, since Python does not have a built-in Poisson
    # random number generate, we use one from numpy.
    # N.B.: this is not how we would use this generator if we were
    # trying to write efficient code.
    generated_counts = [np_rng.poisson(bin_mean) for bin_mean in current_means]
    return generated_counts

def negative_log_likelihood(a, b, c, d, mass, delta):
    """Calculate the negative log likelihood for a given set of parameters."""
    nll = 0  # Initialize negative log likelihood
    for k in range(1, N_b + 1): # Iterate through each bin in the energy spectrum
        mu = poisson_mean_model(a, b, c, d, mass, delta, k)  # Calculate the expected mean
        observed = OBS[k - 1]  # Get the observed count
        # Add the Poisson probability (log-form for numerical stability)
        nll += -mu + observed * math.log(mu) - math.log(math.factorial(observed))
    return -nll


def fit_pseudoexperiment(data):
    """Find the minimum of the negative log likelihood given pseudoexperiment data."""
    # Initialize parameters
    best_a, best_b, best_c, best_d, best_mass, best_delta = 0, 0, 0, 0, 0, 0
    best_nll = float('inf')  # Start with a very high NLL
    it = 0
    # Grid search over parameters
    for a in [A - dA, A, A + dA]:
        for b in [B - dB, B, B + dB]:
            for c in [C - dC, C, C + dC]:
                for d in [D - dD, D, D + dD]:
                    for mass in range(5, 20):  # Assume mass is an integer for simplicity
                        for delta in [1 - 0.1, 1, 1 + 0.1]:  # Assume delta can vary by 0.1
                            nll = negative_log_likelihood(a, b, c, d, mass, delta)
                            # If this NLL is better (lower), update the best parameters
                            if nll < best_nll:
                                best_a, best_b, best_c, best_d, best_mass, best_delta = a, b, c, d, mass, delta
                                best_nll = nll
    # Return the best parameters
    print(f"total it: {it}")
    return best_a, best_b, best_c, best_d, best_mass, best_delta


def generate_one_lambda(m, Delta):
    """Generate a single value of lambda, by generating and fitting one
    pseudoexperiment.
    """
    pe = generate_pseudoexperiment(m, Delta)
    lambda_pe = fit_pseudoexperiment(pe)
    return lambda_pe


if __name__ == "__main__":
    # Pick a spot in our parameter space
    m, Delta = 8.0, 2.0

    lambda_pe = generate_one_lambda(m, Delta)
    print(f"lambda: {lambda_pe}")
