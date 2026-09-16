import numpy as np
import torch

from scipy import linalg
from numpy.typing import NDArray
from torch.types import Tensor

def compute_statistics(feats: NDArray) -> tuple[NDArray, NDArray]:
    """
    Compute the mean and covariance of the features.
    """
    mu = np.mean(feats, axis=0)
    sig = np.cov(feats, rowvar=False)
    return mu, sig

def _frechet_distance(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, eps: float = 1e-6
) -> float:
    """
    Numpy implementation of the Frechet Distance, taken from clean-fid
    -> https://github.com/GaParmar/clean-fid/blob/main/cleanfid/fid.py

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) computed using the standard estimator:

            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Danica J. Sutherland.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    mu1, mu2 : (p,) array-like
        Sample means of the two distributions.

    sigma1, sigma2 : (p, p) array-like
        Empirical covariance matrices of the two distributions.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    float
        Fréchet distance estimate.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def _efficient_frechet_distance(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, eps: float = 1e-6
) -> float:
    """
    Using a nice trick from /u/donshell on reddit
    https://old.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
    > Recall that 1) trace(A) equals the sum of A's eigenvalues and 2) the eigenvalues of sqrt(A) are
    > the square-roots of the eigenvalues of A. Then trace(sqrt(A)) is the sum of square-roots of the
    > eigenvalues of A. Hence, instead of the full square-root we can only compute the eigenvalues of A.

    Efficient implementation that gives the same result as _frechet_distance, but is faster and more memory efficient.
    """
    sqrt_trace = np.real(linalg.eigvals(sigma1 @ sigma2) ** 0.5).sum()
    return ((mu1 - mu2) ** 2).sum() + sigma1.trace() + sigma2.trace() - 2 * sqrt_trace


def frechet_distance(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, efficient: bool = True
) -> float:
    """
    Wrapper function for the Frechet distance that uses the efficient implementation.
    """
    if efficient:
        return _efficient_frechet_distance(mu1, sigma1, mu2, sigma2)
    else:
        return _frechet_distance(mu1, sigma1, mu2, sigma2)


def frechet_distance_torch(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor
) -> Tensor:
    """
    PyTorch implementation of the Fréchet distance

    Using a nice trick from /u/donshell on reddit
    https://old.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/
    > Recall that 1) trace(A) equals the sum of A's eigenvalues and 2) the eigenvalues of sqrt(A) are
    > the square-roots of the eigenvalues of A. Then trace(sqrt(A)) is the sum of square-roots of the
    > eigenvalues of A. Hence, instead of the full square-root we can only compute the eigenvalues of A.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    mu1, mu2 : (p,) array-like
        Sample means of the two distributions.

    sigma1, sigma2 : (p, p) array-like
        Empirical covariance matrices of the two distributions.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    float
        Fréchet distance estimate.
    """
    mean_term = (mu1 - mu2).square().sum(dim=-1)
    tr_cov = sigma1.trace() + sigma2.trace()
    tr_covmean = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return mean_term + tr_cov - 2 * tr_covmean


def fid_from_feats(feats1: NDArray, feats2: NDArray) -> float:
    mu1, sig1 = compute_statistics(feats1)
    mu2, sig2 = compute_statistics(feats2)
    return frechet_distance(mu1, sig1, mu2, sig2)


def fid_from_feats_torch(feats1: Tensor, feats2: Tensor) -> Tensor:
    mu1 = feats1.mean(dim=0)
    mu2 = feats2.mean(dim=0)

    # Match numpy's np.cov(rowvar=False)
    def cov(x):
        x_centered = x - x.mean(dim=0)
        return (x_centered.t() @ x_centered) / (x.shape[0] - 1)

    sig1 = cov(feats1)
    sig2 = cov(feats2)

    return frechet_distance_torch(mu1, sig1, mu2, sig2)


####################### TESTS ###############################


def _compare_case(name, x1_np, x2_np, atol=1e-6, rtol=1e-6):
    x1_t = torch.from_numpy(x1_np).to(torch.float64)
    x2_t = torch.from_numpy(x2_np).to(torch.float64)

    fid_np = fid_from_feats(x1_np, x2_np)
    fid_t = fid_from_feats_torch(x1_t, x2_t).item()

    abs_err = abs(fid_np - fid_t)
    rel_err = abs_err / max(abs(fid_np), 1.0)

    print(f"\n=== {name} ===")
    print(f"NumPy : {fid_np:.12f}")
    print(f"Torch : {fid_t:.12f}")
    print(f"Abs err: {abs_err:.3e}")
    print(f"Rel err: {rel_err:.3e}")

    np.testing.assert_allclose(fid_np, fid_t, atol=atol, rtol=rtol)

    print("✓ PASS")


def test_fid_torch_matches_numpy():
    np.random.seed(0)
    torch.manual_seed(0)

    settings = [
        (32, 512),
        (512, 2000),
        (1024, 10000),
    ]

    for d, n in settings:
        print("=" * 80)
        print(f"Testing d={d}, n={n}")
        print("=" * 80)

        #
        # Case 1: Identical normal gaussians
        #
        mean = np.zeros(d)
        cov = np.eye(d)

        x1 = np.random.multivariate_normal(mean, cov, size=n)
        x2 = np.random.multivariate_normal(mean, cov, size=n)

        _compare_case(f"Identical gaussians (d={d}, n={n})", x1, x2)

        #
        # Case 2: Different diagonal-covariance Gaussians
        #
        mean1 = np.linspace(0.0, 1.0, d)
        mean2 = np.linspace(-0.5, 0.5, d)

        cov1 = np.diag(np.linspace(0.5, 2.0, d))
        cov2 = np.diag(np.linspace(1.0, 3.0, d))

        x1 = np.random.multivariate_normal(mean1, cov1, size=n)
        x2 = np.random.multivariate_normal(mean2, cov2, size=n)

        _compare_case(f"Diagonal covariance (d={d}, n={n})", x1, x2)

        #
        # Case 3: Different Toeplitz-covariance Gaussians
        #
        mean1 = np.sin(np.linspace(0, np.pi, d))
        mean2 = np.cos(np.linspace(0, np.pi, d))

        cov1 = linalg.toeplitz([0.2**i for i in range(d)])
        cov2 = linalg.toeplitz([0.8**i for i in range(d)])

        x1 = np.random.multivariate_normal(mean1, cov1, size=n)
        x2 = np.random.multivariate_normal(mean2, cov2, size=n)

        _compare_case(f"Toeplitz covariance (d={d}, n={n})", x1, x2)

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_fid_torch_matches_numpy()
