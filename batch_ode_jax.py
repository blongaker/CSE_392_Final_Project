import jax
import jax.numpy as jnp
from diffrax import diffeqsolve, ODETerm, Kvaerno3, Dopri5, PIDController
import time
from scipy.integrate import solve_ivp
import numpy as np

# Definition of Lorenz system
def lorenz96_jax(t, x, args):
    F = args[0]
    x_minus_1 = jnp.roll(x, 1)
    x_minus_2 = jnp.roll(x, 2)
    x_plus_1 = jnp.roll(x, -1)
    dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
    return dxdt

# Numpy version to test with scipy
def lorenz96_np(t, x, F):
    x_minus_1 = np.roll(x, 1)
    x_minus_2 = np.roll(x, 2)
    x_plus_1 = np.roll(x, -1)
    dxdt = (x_plus_1 - x_minus_2) * x_minus_1 - x + F
    return dxdt

# Define flowmap
def lorenz96_flowmap_jax(y0, param):
    rhs = ODETerm(lorenz96_jax)
    solver = Dopri5()
    # solver = Kvaerno3()
    controller = PIDController(rtol=1e-6, atol=1e-9)
    sol = diffeqsolve(
        rhs,
        solver,
        t0=0,
        t1=10,
        dt0=0.05,
        y0=y0,
        args=param,
        stepsize_controller=controller
    )
    return sol.ys


print(f'Default device for jax: {jax.default_backend()}')
devices = jax.devices()
print(f'Available devices: {devices}')

# Create a batch of initial conditions
key = jax.random.PRNGKey(42)
N = 100000
d = 5
batch_y0 = jax.random.uniform(key, (N, d), minval=-5.0, maxval=5.0)
batch_p = jax.random.uniform(key, (N, 1), minval=1.0, maxval=5.0)

# 4. Use jax.vmap to batch the solve
# in_axes=(None, None, None, None, None, 0, None) maps over y0 (the 6th argument)
batch_solve = jax.vmap(lorenz96_flowmap_jax, in_axes=(0,0))
fast_batch_solve = jax.jit(batch_solve)

warmup_y0 = jnp.array([
    [1,2,3,4,5],
    [1,2,3,4,5]
])
warmup_p = jnp.array([
    [1],
    [2],
])
fast_batch_solve(warmup_y0, warmup_p)

start_time = time.perf_counter()
results = fast_batch_solve(batch_y0, batch_p)
end_time = time.perf_counter()
time_elapsed = end_time - start_time


print(f'Result shape: {results.shape}')


# Test the exact same thing with scipy
# sol_scipy = solve_ivp(lambda t, y: lorenz96_np(t, y, np.array(batch_p[test_row])), [0, 10], np.array(batch_y0[test_row]), method='RK45', rtol=1e-6)
# print(f'Result with scipy: {sol_scipy.y[:,-1]}')