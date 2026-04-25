import cupy as cp
from solvers import ODESolver
from right_hand_sides import DifferentialEquation


class BatchedODESolver:

    def __init__(self, diffeq: DifferentialEquation, t: float, y: cp.ndarray, params: cp.ndarray, t_eval, solver: ODESolver):
        
        # Record dimensions and make sure shapes match
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
        self.t_eval = t_eval
        self.solver = solver

        # Extract the right-hand side kernel
        ptr_rhs = diffeq.rhs_module.get_global('ptr_rhs')
        self.rhs_addr = cp.ndarray(shape=(1,), dtype=cp.uint64, memptr=ptr_rhs)

        # All kernels already compiled? Just need the pointer to the rhs function
        self.timestep_kernel = solver.ode_solver_raw_kernel.get_function('timestep_kernel')


    def step(self, dt: float, threads: int = 256):
        # Launch the Kernel for a single step
        blocks = (self.n_odes + threads - 1) // threads
        self.timestep_kernel((blocks,), (threads,), (rhs_addr[0], self.t, dt, self.y, self.params))



if __name__ == '__main__':

    # Print input
    print('Input (first two states):')
    print(y_gpu[:2])

    # Define right hand side of solver
    ptr_predator_prey = rhs_module.get_global('ptr_predator_prey')
    # We need to extract the actual address value to pass to the kernel
    # We create a 1-element array to interpret the memory at that global location
    rhs_addr = cp.ndarray(shape=(1,), dtype=cp.uint64, memptr=ptr_predator_prey)

    # Launch the Kernel for a single step
    threads = 256
    blocks = (n_odes + threads - 1) // threads
    kernel((blocks,), (threads,), (rhs_addr[0], 0, 0.05, y_gpu, params, n_odes, n_vars, n_params))

    # Output
    print('Output (first two states):')
    print(y_gpu[:2])






