import cupy as cp
import numpy as np
from solvers import ODESolver, ForwardEulerSolver
from right_hand_sides import DifferentialEquation


class BatchedODESolver:

    def __init__(
            self,
            diffeq: DifferentialEquation,
            t: float,
            y: cp.ndarray,
            params: cp.ndarray,
            solver: ODESolver
        ):

        n_odes, n_vars = y.shape
        n_odes_2, n_params = params.shape
        assert n_odes == n_odes_2

        # Save everything
        self.n_odes = n_odes
        self.n_vars = n_vars
        self.n_params = n_params
        self.diffeq = diffeq
        self.t      = t
        self.y      = y
        self.params = params
        self.solver = solver

        # Extract the right-hand side kernel
        ptr_rhs = diffeq.rhs_module.get_global('ptr_rhs')
        self.rhs_addr = cp.ndarray(shape=(1,), dtype=cp.uint64, memptr=ptr_rhs)

        # All kernels already compiled? Just need the pointer to the rhs function
        self.timestep_kernel = solver.ode_solver_raw_kernel.get_function('timestep_kernel')


    def step(self, dt: float, threads: int = 256):
        # Launch the Kernel for a single step
        blocks = (self.n_odes + threads - 1) // threads
        self.timestep_kernel((blocks,), (threads,), (self.rhs_addr[0], np.float32(self.t), np.float32(dt), self.y, self.params))
        self.t += dt



if __name__ == '__main__':

    # Generate initial condition data
    n_odes = 1000000
    n_vars = 2
    n_params = 4
    y = cp.random.uniform(-1, 1, (n_odes, n_vars), dtype=cp.float32) # type: ignore
    p = cp.random.uniform( 0, 1, (n_odes, n_params), dtype=cp.float32) # type: ignore

    # Print input
    print('Parameters:')
    print(p)
    print('t=0 (first two states):')
    print(y[:2])

    # Batched ODE Solve
    lotka_volterra = DifferentialEquation(
        cuda_code = r'''
            dydt[0] =  p[0] * y[0] - p[1] * y[0] * y[1];
            dydt[1] = -p[3] * y[1] + p[3] * y[0] * y[1];
        '''
    )
    forward_euler_solver = ForwardEulerSolver(n_odes, n_vars, n_params)
    batched_ode_solver = BatchedODESolver(lotka_volterra, 0, y, p, forward_euler_solver)

    # Launch the Kernel for a single step
    batched_ode_solver.step(0.01)

    # Output
    print('t=0.01 (first two states):')
    print(y[:2])

    # One more time for good measure
    batched_ode_solver.step(0.01)
    print('t=0.02 (first two states):')
    print(y[:2])
