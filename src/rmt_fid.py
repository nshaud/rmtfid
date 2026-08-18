import numpy as np
import torch

from scipy import linalg

from torch.types import Tensor
from numpy.typing import ArrayLike, NDArray


def rmt_frechet_distance_analytical(
    mu1: NDArray, sigma1: NDArray, mu2: NDArray, sigma2: NDArray, n: int
) -> float:
    """
    Random Matrix Theory (RMT)-corrected Fréchet distance between two Gaussian distributions.

    This implementation follows the estimator proposed in:

        Tiomoko, A. & Couillet, R.
        "Random Matrix-Improved Estimation of the Wasserstein Distance
        between two Centered Gaussian Distributions"
        https://github.com/maliktiomoko/RMTWasserstein

    Note that it assumes that sigma1 and sigma2 are empirical covariance matrices estimated
    from the same number of samples n.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    mu1, mu2 : (p,) array-like
        Sample means of the two distributions.

    sigma1, sigma2 : (p, p) array-like
        Empirical covariance matrices of the two distributions.

    n : int
        Number of samples used to estimate each covariance matrix.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    rmt_wasserstein : float
        RMT-corrected squared Wasserstein (Fréchet) distance estimate.
    """
    mean_difference = mu1 - mu2

    eps = 1e-12
    offset = np.eye(sigma1.shape[0]) * eps

    # Eigenvalues of covariance product sigma1 @ sigma2
    # ------------------------------
    # lambda=sort(eig(hatC1*hatC2));
    # ------------------------------
    lambda_ = np.sort(np.real(np.linalg.eigvals(sigma1 @ sigma2 + offset)))
    sqrt_lambda = np.sqrt(lambda_).reshape(-1, 1)

    # Rank-1 spectral correction matrix
    # --------------------------------------------------------------------
    # eta=sort(real(eig(diag(lambda)-(1/n1)*sqrt(lambda)*sqrt(lambda)')));
    # --------------------------------------------------------------------
    delta = sqrt_lambda @ sqrt_lambda.transpose()

    eta = np.sort(np.real(np.linalg.eigvals(np.diag(lambda_) - (1 / n) * delta)))

    # Covariance contribution (RMT-corrected)
    # ----------------------------------------------------------------------------
    # est=(1/p)*trace(hatC1+hatC2)-2*(sum(sqrt(lambda))-sum(sqrt(zeta)))*(2*n1/p);
    # ----------------------------------------------------------------------------
    covariance_term = np.trace(sigma1 + sigma2) - 2 * (
        np.sum(sqrt_lambda) - np.sum(np.sqrt(eta))
    ) * (2 * n)

    # Final RMT-Fréchet distance
    rmt_wasserstein = mean_difference.dot(mean_difference) + covariance_term

    return rmt_wasserstein


def rmt_fid_from_feats(feats1: NDArray, feats2: NDArray) -> float:
    assert len(feats1) == len(feats2)
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return rmt_frechet_distance_analytical(mu1, sig1, mu2, sig2, len(feats1))


def rmt_frechet_distance_analytical_pytorch(
    mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor, n: int
) -> Tensor:
    """
    Random Matrix Theory (RMT)-corrected Fréchet distance between two Gaussian distributions.

    -> PyTorch implementation which supports autograd.

    This implementation follows the estimator proposed in:

        Tiomoko, A. & Couillet, R.
        "Random Matrix-Improved Estimation of the Wasserstein Distance
        between two Centered Gaussian Distributions"
        https://github.com/maliktiomoko/RMTWasserstein

    Note that it assumes that sigma1 and sigma2 are empirical covariance matrices estimated
    from the same number of samples n.

    --------------------------------------------------------------------------
    Parameters
    --------------------------------------------------------------------------
    mu1, mu2 : (d,) array-like
        Sample means of the two distributions.

    sigma1, sigma2 : (d, p) array-like
        Empirical covariance matrices of the two distributions.

    n : int
        Number of samples used to estimate each covariance matrix.

    --------------------------------------------------------------------------
    Returns
    --------------------------------------------------------------------------
    rmt_wasserstein : float
        RMT-corrected squared Wasserstein (Fréchet) distance estimate.
    """
    mean_difference = mu1 - mu2

    # Small perturbation for numerical stability, might be unecessary
    eps = 1e-12
    d = sigma1.shape[0]
    offset = torch.eye(d, device=sigma1.device, dtype=sigma1.dtype) * eps

    # Eigenvalues of covariance product sigma1 @ sigma2
    # ------------------------------
    # lambda=sort(eig(hatC1*hatC2));
    # ------------------------------
    lambda_ = torch.sort(torch.linalg.eigvals(sigma1 @ sigma2 + offset).real).values
    lambda_ = torch.clamp(lambda_, min=0.0)

    sqrt_lambda = torch.sqrt(lambda_).reshape(-1, 1)  # (d, 1)

    # Rank-1 spectral correction matrix
    # --------------------------------------------------------------------
    # eta=sort(real(eig(diag(lambda)-(1/n1)*sqrt(lambda)*sqrt(lambda)')));
    # --------------------------------------------------------------------
    delta = sqrt_lambda @ sqrt_lambda.t()  # outer product

    eta = torch.sort(
        torch.linalg.eigvals(torch.diag(lambda_) - (1.0 / n) * delta).real
    ).values
    eta = torch.clamp(eta, min=0.0)
    sqrt_eta = torch.sqrt(eta)

    # Covariance contribution (RMT-corrected)
    # ----------------------------------------------------------------------------
    # est=(1/p)*trace(hatC1+hatC2)-2*(sum(sqrt(lambda))-sum(sqrt(zeta)))*(2*n1/p);
    # ----------------------------------------------------------------------------
    covariance_term = torch.trace(sigma1 + sigma2) - 2 * (
        torch.sum(sqrt_lambda) - torch.sum(sqrt_eta)
    ) * (2 * n)

    # Final RMT-Fréchet distance
    rmt_wasserstein = mean_difference.dot(mean_difference) + covariance_term

    return rmt_wasserstein


def rmt_fid_from_feats_torch(feats1: Tensor, feats2: Tensor) -> Tensor:
    mu1 = feats1.mean(dim=0)
    mu2 = feats2.mean(dim=0)

    # Match numpy's np.cov(rowvar=False)
    def cov(x):
        x_centered = x - x.mean(dim=0)
        return (x_centered.t() @ x_centered) / (x.shape[0] - 1)

    sig1 = cov(feats1)
    sig2 = cov(feats2)

    n = feats1.shape[0]
    return rmt_frechet_distance_analytical_pytorch(mu1, sig1, mu2, sig2, n)


#################### TESTS #############################


def _compare_case(name, x1_np, x2_np, atol=1e-6, rtol=1e-6):
    x1_t = torch.from_numpy(x1_np).to(torch.float64)
    x2_t = torch.from_numpy(x2_np).to(torch.float64)

    fid_np = rmt_fid_from_feats(x1_np, x2_np)
    fid_t = rmt_fid_from_feats_torch(x1_t, x2_t).item()

    abs_err = abs(fid_np - fid_t)
    rel_err = abs_err / max(abs(fid_np), 1.0)

    print(f"\n=== {name} ===")
    print(f"NumPy : {fid_np:.12f}")
    print(f"Torch : {fid_t:.12f}")
    print(f"Abs err: {abs_err:.3e}")
    print(f"Rel err: {rel_err:.3e}")

    np.testing.assert_allclose(fid_np, fid_t, atol=atol, rtol=rtol)

    print("✓ PASS")


def test_rmt_fid_torch_matches_numpy():
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
    test_rmt_fid_torch_matches_numpy()
