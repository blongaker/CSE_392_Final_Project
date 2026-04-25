import cupy as cp
from solvers import solvers_module
from right_hand_sides import rhs_module

def batched_ode_solve(
    fun: function,
    t_span,
    y0,
    params,
    t_eval=None
):
    """
    A batched ODE solver.

    Args:
        fun (function): The right-hand side function. Takes a single state and parameter vector.
        t_span (_type_): The time integration range.
        y0 (_type_): The matrix of initial states. Each row is a single state.
        params (_type_): The matrix of parameters. Each row is a single parameter vector.
        t_eval (None, optional): The times to return for the solver. If None, then we only return the state at the final time. Otherwise, we return a tensor.
    """
    return 67


if __name__ == '__main__':

    kernel = solvers_module.get_function('forward_euler_kernel')

    # Load data
    n_odes = 100000
    n_vars = 2
    n_params = 4
    y_gpu = cp.random.uniform(-1, 1, (n_odes, n_vars))
    params = cp.random.uniform(0, 1, (n_odes, n_params))
    t_span = [0, 1]

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






