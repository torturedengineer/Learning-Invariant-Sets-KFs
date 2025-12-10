import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.linalg import solve
import jax.scipy.optimize
import numpy as np
import logging

# Try to import JAXOPT for L-BFGS (matches paper memory efficiency)
try:
    import jaxopt
    HAS_JAXOPT = True
except ImportError:
    HAS_JAXOPT = False
    logging.warning("JAXOPT not found. Falling back to standard BFGS. Install jaxopt for L-BFGS.")

# --- 1. Universal Kernel (Handles Both Cases) ---

@jit
def universal_kernel(x1, x2, alphas, base_sigmas, log_scales):
    """
    Computes the sparse kernel sum.
    Note: alphas are squared (alphas[i]**2) to ensure positive semi-definiteness.
    """
    # A. Apply Anisotropy
    scales = jnp.exp(log_scales)
    x1_scaled = x1 / scales[None, :]
    x2_scaled = x2 / scales[None, :]
    
    # B. Squared Euclidean Distance
    diff = x1_scaled[:, None, :] - x2_scaled[None, :, :]
    dist_sq = jnp.sum(diff**2, axis=-1)
    
    # C. Sparse Dictionary Sum
    K_sum = jnp.zeros_like(dist_sq)
    for i in range(len(base_sigmas)):
        K_i = jnp.exp(-0.5 * dist_sq / (base_sigmas[i]**2))
        K_sum = K_sum + (alphas[i]**2) * K_i
        
    return K_sum

# --- 2. Modified Hausdorff Loss ---
@jit
def modified_hausdorff_loss(set_a, set_b):
    diff = set_a[:, None, :] - set_b[None, :, :]
    dists_sq = jnp.sum(diff**2, axis=-1)
    return 0.5 * (jnp.mean(jnp.min(dists_sq, axis=1)) + jnp.mean(jnp.min(dists_sq, axis=0)))

# --- 3. The Hybrid Model Class ---

class SparseKernelFlowsModel:
    def __init__(self, regularization=1e-4, mhd_weight=0.1, l1_weight=1e-3, is_anisotropic=True):
        self.reg = regularization
        self.mhd_weight = mhd_weight 
        self.l1_weight = l1_weight
        self.is_anisotropic = is_anisotropic
        self.base_sigmas = jnp.logspace(-1, 2, 10)
        
        self.params = {
            "alphas": None, "log_scales": None, "weights": None, "X_train": None
        }
        
    def fit(self, X, Y, dt=0.01):
        # Data Shuffling
        X_np, Y_np = np.array(X), np.array(Y)
        perm = np.random.permutation(len(X_np))
        X = jnp.array(X_np[perm])
        Y = jnp.array(Y_np[perm])

        N, D = X.shape
        n_kernels = len(self.base_sigmas)
        
        # Init parameters
        init_alphas = jnp.ones(n_kernels) / n_kernels
        init_log_scales = jnp.zeros(D)
        
        if self.is_anisotropic:
            x0 = jnp.concatenate([init_alphas, init_log_scales])
        else:
            x0 = init_alphas
            
        # Split Data
        split = int(N * 0.5)
        X_tr, Y_tr = X[:split], Y[:split]
        X_val, Y_val = X[split:], Y[split:]
        sigmas = self.base_sigmas

        # --- Objective Function ---
        def objective(params_flat):
            if self.is_anisotropic:
                alphas = params_flat[:n_kernels]
                scales = params_flat[n_kernels:]
            else:
                alphas = params_flat
                scales = jnp.zeros(D)

            # Training
            K_train = universal_kernel(X_tr, X_tr, alphas, sigmas, scales)
            w_sub = solve(K_train + self.reg * jnp.eye(len(X_tr)), Y_tr, assume_a='pos')
            
            # Validation
            K_cross = universal_kernel(X_val, X_tr, alphas, sigmas, scales)
            Y_pred = jnp.dot(K_cross, w_sub)
            
            # Loss
            mse = jnp.mean((Y_val - Y_pred)**2)
            loss_dyn = mse / (jnp.mean(Y_val**2) + 1e-6)
            loss_mhd = modified_hausdorff_loss(X_val, X_val + Y_pred * dt)
            loss_sparse = jnp.sum(jnp.abs(alphas**2))
            
            return (1.0 - self.mhd_weight) * loss_dyn + self.mhd_weight * loss_mhd + self.l1_weight * loss_sparse

        # --- Optimization Step ---
        if HAS_JAXOPT:
            # FIX: Use LBFGS (Unconstrained) instead of LBFGSB (Bounded)
            # This avoids the "missing bounds" error while maintaining L-BFGS efficiency
            solver = jaxopt.LBFGS(fun=objective, maxiter=100)
            res = solver.run(x0)
            best_params = res.params
        else:
            res = jax.scipy.optimize.minimize(fun=objective, x0=x0, method='BFGS', options={'maxiter': 100})
            best_params = res.x

        # Unpack & Final Fit
        if self.is_anisotropic:
            best_alphas = best_params[:n_kernels]
            best_scales = best_params[n_kernels:]
        else:
            best_alphas = best_params
            best_scales = jnp.zeros(D)

        final_K = universal_kernel(X, X, best_alphas, sigmas, best_scales)
        self.params["weights"] = solve(final_K + self.reg * jnp.eye(N), Y, assume_a='pos')
        self.params["alphas"] = best_alphas
        self.params["log_scales"] = best_scales
        self.params["X_train"] = X
        
        return best_alphas, jnp.exp(best_scales)

    def predict(self, x_new):
        if x_new.ndim == 1: x_new = x_new[None, :]
        K_vec = universal_kernel(x_new, self.params["X_train"], self.params["alphas"], 
                                 self.base_sigmas, self.params["log_scales"])
        return np.array(jnp.dot(K_vec, self.params["weights"])[0])