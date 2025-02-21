import numpy as np
from scipy.stats import norm

def generate_bimodal_prior(b_values, mean_b1, std_b1, mean_b2, std_b2, weight_b1=1, weight_b2=1):
    """
    Generate a bimodal prior distribution by combining two Gaussian distributions with weights.

    Parameters:
    - b_values: Array of b values where the prior will be evaluated.
    - mean_b1: Mean of the first Gaussian distribution.
    - std_b1: Standard deviation of the first Gaussian distribution.
    - mean_b2: Mean of the second Gaussian distribution.
    - std_b2: Standard deviation of the second Gaussian distribution.
    - weight_b1: Weight for the first Gaussian (higher value increases its importance).
    - weight_b2: Weight for the second Gaussian (higher value increases its importance).

    Returns:
    - bimodal_prior: The combined and normalized bimodal prior distribution.
    """
    # Create two Gaussian distributions
    prior1 = norm.pdf(b_values, loc=mean_b1, scale=std_b1)
    prior2 = norm.pdf(b_values, loc=mean_b2, scale=std_b2)

    # Apply weights to the distributions
    prior1 *= weight_b1
    prior2 *= weight_b2

    # Combine the two distributions to create a bimodal distribution
    bimodal_prior = prior1 + prior2

    # Normalize the combined distribution so it sums to 1
    bimodal_prior /= np.trapz(bimodal_prior,b_values)

    return bimodal_prior
