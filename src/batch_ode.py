import cupy as cp
import numpy as np
from solvers import ODESolver, ForwardEulerSolver


class BatchedODESolver:

    def __init__(
            self,
            solver: ODESolver,
            t: float,
            y: cp.ndarray,
            params: cp.ndarray,
        ):

        n_odes, n_vars = y.shape
        n_odes_2, n_params = params.shape
        assert n_odes == n_odes_2

        # Save everything
        self.n_odes = n_odes
        self.n_vars = n_vars
        self.n_params = n_params
        self.t      = t
        self.y      = y
        self.params = params
        self.solver = solver

        # Compile the kernel function
        self.timestep_kernel = solver.ode_solver_raw_module.get_function('timestep_kernel')


    def step(self, dt: float, threads: int = 256):
        # Launch the Kernel for a single step
        blocks = (self.n_odes + threads - 1) // threads
        self.timestep_kernel((blocks,), (threads,), (np.float32(self.t), np.float32(dt), self.y, self.params))
        cp.cuda.Device().synchronize()
        self.t += dt



if __name__ == '__main__':

    # LORENZ96 EXAMPLE

    # Generate initial condition data
    n_odes = 1000000
    n_vars = 5
    n_params = 1
    y = cp.random.uniform(-5, 5, (n_odes, n_vars), dtype=cp.float32) # type: ignore
    p = cp.random.uniform( 1, 5, (n_odes, n_params), dtype=cp.float32) # type: ignore

    # Print input
    print('Parameters:')
    print(p)
    print('t=0 (first two states):')
    print(y[:2])

    # Batched ODE Solve
    lorenz96 = '''
    dydt[0] = (y[1] - y[3]) * y[4] - y[0] + p[0];
    dydt[1] = (y[2] - y[4]) * y[0] - y[1] + p[0];
    dydt[2] = (y[3] - y[0]) * y[1] - y[2] + p[0];
    dydt[3] = (y[4] - y[1]) * y[2] - y[3] + p[0];
    dydt[4] = (y[0] - y[2]) * y[3] - y[4] + p[0];
    '''

    forward_euler_solver = ForwardEulerSolver(lorenz96, n_odes, n_vars, n_params)
    batched_ode_solver = BatchedODESolver(forward_euler_solver, 0, y, p)

    # Launch the Kernel for a single step
    batched_ode_solver.step(0.01)

    # Output
    print('t=0.01 (first two states):')
    print(y[:2])

    # One more time for good measure
    batched_ode_solver.step(0.01)
    print('t=0.02 (first two states):')
    print(y[:2])
