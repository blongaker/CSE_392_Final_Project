import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def lorenz96(t, X, F):
    """
    Computes the RHS of Lorenz 96 for a matrix of states.
    
    Parameters:
    X : ndarray
        Matrix of shape (M, N) where each row is a state.
    F : float
        Forcing constant.
    """
    
    x_plus_1 = np.roll(X, -1, axis=1)  # x_{i+1}
    x_minus_2 = np.roll(X, 2, axis=1)  # x_{i-2}
    x_minus_1 = np.roll(X, 1, axis=1)  # x_{i-1}
    
    return (x_plus_1 - x_minus_2) * x_minus_1 - X + F


def lorenz96_wrapper(t, x, F, N, d):
    return lorenz96(t, x.reshape(N, d), F).flatten()


def lorenz96_single(t, x, F):
    return (np.roll(x, -1) - np.roll(x, 2)) * np.roll(x, 1) - x + F


# Constant time step
def forward_euler(t0: float, dt: float, tf: float, fun, y0: np.ndarray):
    t_curr = t0
    y_curr = y0
    while t_curr < tf:
        y_curr += dt * fun(t_curr, y_curr)
        t_curr += dt
    return y_curr


def rk23_batched(t0, dt0, tf, fun, y0: np.ndarray, tol=1e-2):
    
    # Initialize
    N, d = y0.shape
    t_curr = np.full((N,1),t0)
    y_curr = y0
    dt_curr = np.full((N,1), dt0)

    # Loop until every time stepper is done
    while (t_curr < tf).any():

        # Compute stages
        k1 = fun(t_curr, y_curr)
        k2 = fun(t_curr + 0.5 * dt_curr, y_curr + 0.5 * dt_curr * k1)
        k3 = fun(t_curr + 0.75 * dt_curr, y_curr + 0.75 * dt_curr * k2)

        # Third order output
        y_next_pert = dt_curr * (2 * k1 + 3 * k2 + 4 * k3) / 9
        k4 = fun(t_curr + dt_curr, y_curr + y_next_pert)

        # Compute error and update accordingly
        error = dt_curr * (-5 * k1 + 6 * k2 + 8 * k3 - 9 * k4)
        error_norm = np.linalg.norm(error, axis=1)
        step_acceptable = error_norm < tol
        y_curr += np.broadcast_to(step_acceptable[:, np.newaxis], (N,d)) * y_next_pert

        # Update time step sizes
        stepping_complete = (t_curr + dt_curr) >= tf
        error_norm[stepping_complete.flatten()] = 1
        t_curr[step_acceptable] += dt_curr[step_acceptable]
        dt_curr *= 0.9 * (tol / (error_norm[:, np.newaxis])) ** (1/3)
        dt_curr = np.minimum(dt_curr, tf - t_curr)


    return y_curr



def lorenz96_sim_cpu_serial(method='forward_euler', N=10000, d=5, dt=0.01, print_output=False):
    
    # Generate initial conditions and parameter data
    np.random.seed(67)
    y0 = np.random.uniform(-5, 5, (N, d))
    p = np.random.uniform(1, 5, (N, 1))
    tspan = [0.0, 1.0]

    # Advance time step
    start_time = time.perf_counter()
    if method == 'forward_euler': 
        yf = forward_euler(tspan[0], dt, tspan[1], lambda t,x: lorenz96(t,x,p), y0.copy())
    elif method == 'rk23_batched':
        yf = rk23_batched(tspan[0], dt, tspan[1], lambda t,x: lorenz96(t,x,p), y0.copy())

    end_time = time.perf_counter()

    # Compare to ode23
    # start_time = time.perf_counter()
    # sol = solve_ivp(lambda t,x: lorenz96_single(t,x,p[0]), tspan, y0[0], rtol=1e-6, method='RK23')
    # print(f'Scipy solver state at final time: {sol.y[:,-1]}')
    # end_time = time.perf_counter()
    # print(end_time - start_time)
    # print(f'My solver at final time: {yf[0]}')

    time_elapsed = end_time - start_time
    if print_output is True:
        print(f'Total time: {time_elapsed} seconds')

    return time_elapsed


def lorenz96_batched_scaling_test(method='forward_euler', d=5, dt=0.01):
    
    Nvec = 10000 * np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    time_vec = []

    for N in Nvec:
        time = lorenz96_sim_cpu_serial(method, N, d, dt, False)
        time_vec.append(time)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(Nvec, time_vec, '-or')
    ax.set_xlabel('Number of ODEs')
    ax.set_ylabel('Time elapsed (seconds)')
    ax.set_title('Strong Scaling, CPU')
    fig.savefig(f'figs/strong_scaling_{method}_cpu.png')



if __name__ == '__main__':
    # lorenz96_sim_cpu_serial(method='rk23_batched', N=200000, d=5, dt=0.01)
    lorenz96_batched_scaling_test(method='rk23_batched', d=5, dt=0.01)
    # lorenz96_batched_scaling_test(method='forward_euler', d=5, dt=0.01)