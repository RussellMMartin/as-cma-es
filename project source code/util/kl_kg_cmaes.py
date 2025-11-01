
import numpy as np
import matplotlib.pyplot as plt
from math import cos, pi, sqrt, exp, erf, sqrt, log


def Phi(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))

def dkl_gaussian_diag(m1, sigma1_diag, m2, sigma2_diag, eps=1e-12):

    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    s1 = np.maximum(np.asarray(sigma1_diag), eps)
    s2 = np.maximum(np.asarray(sigma2_diag), eps)
    d = m1.size

    # log(det Sigma2 / det Sigma1) = sum(log(s2_j)) - sum(log(s1_j))
    log_det_ratio = np.sum(np.log(s2)) - np.sum(np.log(s1))
    # tr(Sigma2^{-1} Sigma1) = sum( s1_j / s2_j )
    trace_term = np.sum(s1 / s2)
    # Mahalanobis term (m2 - m1)^T Sigma2^{-1} (m2 - m1) = sum( (m2_j - m1_j)^2 / s2_j )
    diff = m2 - m1
    maha = np.sum((diff * diff) / s2)

    return 0.5 * (log_det_ratio - d + trace_term + maha)

def compute_selection_indices(S, mu):

    return np.argsort(S)[:mu]

def compute_cma_mean_and_diag_cov(population, idx_selected, mu):

    selected = population[np.array(idx_selected)]
    if selected.shape[0] == 0:
        raise ValueError("No individuals selected to compute CMA params.")
    m = np.mean(selected, axis=0)

    # sample variance-like formula (paper uses 1/(µ-1) factor)
    diffs = selected - m[None, :]
    sigma_diag = np.sum(diffs * diffs, axis=0) / float(mu - 1)

    return m, sigma_diag

def uniform_single_generation(population, F_func, sigma_noise, mu, N, n0=1, rng=None):

    if rng is None:
        rng = np.random.RandomState(0)

    lam, d = population.shape
    n = np.full(lam, n0, dtype=int)
    S = np.zeros(lam, dtype=float)

    # Initial samples
    for i in range(lam):
        xs = F_func(population[i]) + rng.normal(0.0, sigma_noise, size=n0)
        S[i] = np.mean(xs)

    total_samples = int(np.sum(n))

    while total_samples < N:
        chosen = rng.randint(lam)  # pick a random individual
        x_new = F_func(population[chosen]) + rng.normal(0.0, sigma_noise)
        S[chosen] = (n[chosen] * S[chosen] + x_new) / float(n[chosen] + 1)
        n[chosen] += 1
        total_samples += 1

    idx_final = compute_selection_indices(S, mu)
    m_final, sigma_final = compute_cma_mean_and_diag_cov(population, idx_final, mu)

    return {
        "I_mu_indices": idx_final,
        "S": S,
        "n_counts": n,
        "m_final": m_final,
        "sigma_final": sigma_final
    }
def select_next_candidate_kl_kg(population, S, n, sigma_noise, mu):

    lam, d = population.shape

    # Current selection and CMA params
    idx_current = compute_selection_indices(S, mu)
    m_current, sigma_current = compute_cma_mean_and_diag_cov(population, idx_current, mu)

    V = np.zeros(lam, dtype=float)

    order = np.argsort(S)
    S_sorted = S[order]
    S_mu = S_sorted[mu - 1]
    S_mu_plus1 = S_sorted[mu]

    for i in range(lam):
        # δ_i depends on membership in current top-µ
        if i in idx_current:
            delta_i = S[i] - S_mu_plus1
        else:
            delta_i = S_mu - S[i]

        # γ_i 
        gamma_i = (n[i] + 1) * delta_i + S[i]

        # P_change = 1 - Φ(|γ_i - S_i| / σ)
        arg = abs(gamma_i - S[i]) / float(sigma_noise)
        p_change = 1.0 - Phi(arg)

        # Hypothetical new mean for i after one more sample that would change selection
        S_i_hyp = S[i] + delta_i
        S_hyp = S.copy()
        S_hyp[i] = S_i_hyp

        # Recompute hypothetical selection and CMA params
        idx_hyp = compute_selection_indices(S_hyp, mu)
        m_hyp, sigma_hyp = compute_cma_mean_and_diag_cov(population, idx_hyp, mu)

        # DKL between hypothetical and current CMA distributions
        dkl_val = dkl_gaussian_diag(m_hyp, sigma_hyp, m_current, sigma_current)

        V[i] = p_change * dkl_val

    return int(np.argmax(V))


def kl_kg_single_generation(population, F_func, sigma_noise, mu, N, n0, rng=None):

    
    if rng is None:
        rng = np.random.RandomState(0)

    lam, d = population.shape
    # Initialize per paper: ni = n0 and initial samples
    n = np.full(lam, n0, dtype=int)
    S = np.zeros(lam, dtype=float)

    # INITIALIZE: perform n0 samples of each individual
    for i in range(lam):
        xs = F_func(population[i]) + rng.normal(loc=0.0, scale=sigma_noise, size=n0)
        S[i] = np.mean(xs)

    total_samples = int(np.sum(n))

    while total_samples < N:
        # Decide which candidate to sample next using extracted decision function
        chosen = select_next_candidate_kl_kg(population, S, n, sigma_noise, mu)

        # Perform an actual noisy sample from chosen individual
        x_new = F_func(population[chosen]) + rng.normal(loc=0.0, scale=sigma_noise)
        S[chosen] = (n[chosen] * S[chosen] + x_new) / float(n[chosen] + 1)
        n[chosen] += 1
        total_samples += 1

    # End loop: RETURN final selection Iµ
    idx_final = compute_selection_indices(S, mu)
    m_final, sigma_final = compute_cma_mean_and_diag_cov(population, idx_final, mu)
    return {
        "I_mu_indices": idx_final,
        "S": S,
        "n_counts": n,
        "m_final": m_final,
        "sigma_final": sigma_final
    }
