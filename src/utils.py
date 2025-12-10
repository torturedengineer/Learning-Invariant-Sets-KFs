import numpy as np
import jax.numpy as jnp

def standardize_data(data):
    """
    Standardizes data to mean=0, std=1.
    Returns standardized data, mean, and std for inverse transformation.
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-6  # Add epsilon to prevent divide by zero
    standardized_data = (data - mean) / std
    return standardized_data, mean, std

def get_derivatives(data, dt):
    """
    Approximates time derivatives (velocities) from the trajectory.
    v(t) = (x(t+1) - x(t)) / dt
    """
    return (data[1:] - data[:-1]) / dt

def rk4_integrate(model_predict_fn, x0, dt, steps):
    """
    4th-Order Runge-Kutta Integrator for the Learned Model.
    
    Args:
        model_predict_fn: A function f(x) that returns velocity (dx/dt).
        x0: Initial condition (numpy array).
        dt: Time step.
        steps: Number of integration steps.
        
    Returns:
        trajectory: (steps, D) numpy array.
    """
    D = len(x0)
    trajectory = np.zeros((steps, D))
    trajectory[0] = x0
    x = x0

    # Ensure x is a JAX-compatible array if the model expects it
    for i in range(1, steps):
        # RK4 steps
        k1 = model_predict_fn(x)
        k2 = model_predict_fn(x + 0.5 * dt * k1)
        k3 = model_predict_fn(x + 0.5 * dt * k2)
        k4 = model_predict_fn(x + dt * k3)
        
        # Update
        x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        trajectory[i] = x
        
    return trajectory