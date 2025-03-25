import numpy as np
from scipy.stats import chi2_contingency
import ot
import torch

def chisquaremetric(gen_data, data):
    # gen_data = np.log(gen_data)
    # data = np.log(data)
    # Define bins for both dimensions
    x_bins = np.linspace(min(gen_data[:, 0].min(), data[:, 0].min()), 
                        max(gen_data[:, 0].max(), data[:, 0].max()), 10)
    y_bins = np.linspace(min(gen_data[:, 1].min(), data[:, 1].min()), 
                        max(gen_data[:, 1].max(), data[:, 1].max()), 10)

    # Bin the data in 2D
    observed_gen, _, _ = np.histogram2d(gen_data[:, 0], gen_data[:, 1], bins=[x_bins, y_bins])
    observed_data, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=[x_bins, y_bins])

    # Perform chi-squared test
    chi2_stat, p_value, _, _ = chi2_contingency([observed_gen.flatten()+1e-8, observed_data.flatten()+1e-8])

    return chi2_stat, p_value

def emd(gen_data, data):
    # Define uniform distributions on the point clouds
    p = np.ones((data.shape[0],)) / data.shape[0]  # 1/N for N points
    q = p # np.ones((Y.shape[0],)) / Y.shape[0]

    # Compute the pairwise distance matrix between points in X and Y
    generated_pairwise_distance = ot.dist(gen_data, data, metric='euclidean')
    return(ot.emd2(p, q, generated_pairwise_distance))

def unpack_params(packed_params, p, n):
    """
    Unpack the single tensor back into means, variances, and weights.
    """
    means = []
    idx = 0

    for i in range(p):
        means.append(packed_params[idx:idx + n].tolist())
        idx += n
    # Debugging statements
    # print(f"Unpacked Means: {means}")
    return torch.tensor(means)

def generator_hinge_loss(d_fake):
    """
    Compute the hinge loss for the generator.
    
    Args:
    - d_fake: Discriminator output for fake (generated) samples.
    
    Returns:
    - loss: Generator loss.
    """
    # Generator loss: -mean(D(x_fake))
    return -torch.mean(d_fake)

def discriminator_hinge_loss(d_real, d_fake):
    """
    Compute the hinge loss for the discriminator.
    
    Args:
    - d_real: Discriminator output for real samples.
    - d_fake: Discriminator output for fake (generated) samples.
    
    Returns:
    - loss: Total discriminator loss (for real and fake samples).
    """
    # Real samples loss: max(0, 1 - D(x_real))
    real_loss = torch.mean(torch.relu(1.0 - d_real))
    
    # Fake samples loss: max(0, 1 + D(x_fake))
    fake_loss = torch.mean(torch.relu(1.0 + d_fake))
    
    # Total discriminator loss
    return real_loss + fake_loss