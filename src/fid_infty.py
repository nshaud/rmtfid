import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray

from .fid import fid_from_feats


# Implementation adapted from https://github.com/mchong6/FID_IS_infinity/blob/master/score_infinity.py
def fid_infinity_from_feats(
    reference_feats: NDArray,
    synthetic_feats: NDArray,
    num_points: int = 15,
    min_samples: int = 5000,
    plot: bool = False,
) -> float:
    """
    Calculates the (unbiased) FID infinity by extrapolating the FID values
    at different number of samples.
    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        reference_feats: NDArray
            Features of the real dataset.
            It is expected that the real dataset is larger than the synthetic dataset,
            and significantly so (so that bias from covariance estimation in the real
            dataset is small).
        synthetic_feats: NDArray
            Features of the generated dataset.
            Sampling will happen over this dataset to compute FID_N at different sample sizes.
        num_points: (int)
            Number of FID_N that are computed to fit the 1/N line.
            Default: 15
        min_samples: (int)
            Minimum number of samples to evaluate FID N.
            Default: p + 1, where p is the dimension of the features.
            (original implementation uses 5000, but this is too high for small datasets)
        plot: (bool)
            Set to True to plot the FID_N values and the fitted line with matplotlib.
            Default: False

    --------------------------------------------------------------------------
    Returns:
    --------------------------------------------------------------------------
        fid_infinity: (float)
            The extrapolated FID value at N -> +infinity.

    """
    # Check that there are a least N > min_samples features in both feats1 and feats2
    assert min_samples < min(
        len(reference_feats), len(synthetic_feats)
    ), f"min_samples ({min_samples}) must be less than the number of samples in both feature sets (found {len(reference_feats)}, {len(synthetic_feats)})"

    fid_values = []

    # Choose the number of samples to evaluate FID_N at regular intervals over N
    n_samples = np.linspace(min_samples, len(synthetic_feats), num_points).astype(
        "int32"
    )

    n_samples = np.unique(n_samples)  # Remove duplicates and warn if there are any
    if len(n_samples) < num_points:
        print(
            f"Warning: fewer than {num_points} unique sample sizes found. Using {len(n_samples)} unique sample sizes instead."
        )

    # Evaluate FID_N
    for n in n_samples:
        # sample with replacement
        np.random.shuffle(synthetic_feats)
        feats_n = synthetic_feats[:n]
        fid_n = fid_from_feats(feats_n, reference_feats)
        fid_values.append(fid_n)
    fid_values = np.array(fid_values).reshape(-1, 1)

    # Reshape for linear regression
    n_samples = np.array(n_samples).reshape(-1, 1)
    fid_values = np.array(fid_values).reshape(-1, 1)
    # Fit linear regression in 1/n
    reg = LinearRegression().fit(1 / n_samples, fid_values)
    # Extrapolate FID at n -> +infty samples (i.e. 1/n = 0)
    fid_infinity = reg.predict(np.array([[0]]))[0, 0]

    if plot:
        plt.scatter(1 / n_samples, fid_values)
        X = np.linspace(0, max(1 / n_samples), num=50)
        plt.plot(X, reg.predict(X), color="red")
        plt.show()
    return fid_infinity


def quick_fid_infinity(fid_values, n_samples):
    """
    Quickly computes the FID infinity by fitting a linear regression to the given FID
    values and sample sizes.

    *Caveat: this does not take care of properly sampling the FID_N at different samples
    sizes. It also does not care where the FID values come from. This assumes that the
    FID values are at least somewhat representative of the FID_N values at different
    samples sizes, and reasonably distributed. Use with caution.*

    --------------------------------------------------------------------------
    Parameters:
    --------------------------------------------------------------------------
        fid_values: (array-like)
            The FID values computed at different sample sizes.
        n_samples: (array-like)
            The corresponding sample sizes for the FID values.

    --------------------------------------------------------------------------
    Returns:
    --------------------------------------------------------------------------
        fid_infinity: (float)
            The extrapolated FID value at N -> +infinity.
    """
    # Convert to NumPy and reshape for linear regression
    n_samples = np.array(n_samples).reshape(-1, 1)
    fid_values = np.array(fid_values).reshape(-1, 1)

    # Fit linear regression in 1/n
    reg = LinearRegression().fit(1 / n_samples, fid_values)
    # Extrapolate FID at n = +infinity samples (i.e. 1/n = 0)
    fid_infinity = reg.predict(np.array([[0]]))[0, 0]
    return fid_infinity