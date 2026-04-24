import numpy as np
from clustering import cluster_ics_and_params


def modified_rhs(single_rhs, parameters):
    """
    Takes a right-hand side for an ODE and a list of parameters

    Args:
        single_rhs (function): A right-hand side function that is designed
            to take a single row of parameters and a single state and output the derivative.
        parameters (array): A matrix of parameters. Each row corresponds to different parameters.
    """
    return 67


def batched_ode_solve(
    fun: function,
    t_span,
    y0,
    params,
    method='RK45',
    n_batches=100,
    t_eval=None
):
    """
    A batched ODE solver.

    Args:
        fun (function): The right-hand side function. Takes a single state and parameter vector.
        t_span (_type_): The time integration range.
        y0 (_type_): The matrix of initial states. Each row is a single state.
        params (_type_): The matrix of parameters. Each row is a single parameter vector.
        method (str, optional): Integration method. Defaults to 'RK45'.
        n_batches (int, optional): The number of batches for the GPU solver. Defaults to 100.
        t_eval (None, optional): The times to return for the solver. If None, then we only return the state at the final time. Otherwise, we return a tensor.
    """


    ### --- STEP 1: Clustering --- ###
    # We can do this efficiently on the GPU using CuML k-means.
    sorted_y0, sorted_params, row_starts = cluster_ics_and_params(y0, params, n_batches=100)

    ### --- STEP 2: Build modified right-hand side. --- ###
    # Challenge: can we build the right-hand side that is efficiently vectorized
    #    or do we just need to assume the right-hand-side they give us is easily batched?

    ### --- STEP 3: Loop over batches and send to GPU to solve. --- ###
    # Here we probably just need to do loops. If the batch size is huge we will run out of memory quickly.
    # Plus this is much easier, and can be easily extended to multi-GPU without loss of generality.

    return 67