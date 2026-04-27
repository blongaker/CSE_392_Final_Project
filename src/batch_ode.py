import cupy as cp
import numpy as np
from solvers import ODESolver, ForwardEulerSolver, ForwardEulerSolverMultistep, RK23SolverVarstep
import time
import matplotlib.pyplot as plt


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


    def launch(self, dt: float, threads: int = 256, solver='forward_euler_multistep'):
        # Launch the Kernel for a single step
        blocks = (self.n_odes + threads - 1) // threads
        if solver == 'forward_euler_multistep':
            self.timestep_kernel((blocks,), (threads,), (np.float32(self.t), np.float32(dt), np.float32(100*dt), self.y, self.params))
            self.t = 100 * dt
        elif solver == 'forward_euler_singlestep':
            self.timestep_kernel((blocks,), (threads,), (np.float32(self.t), np.float32(dt), self.y, self.params))
            self.t += dt
        elif solver == 'rk23_multistep':
            self.timestep_kernel((blocks,), (threads,), (np.float32(self.t), np.float32(dt), np.float32(100*dt), self.y, self.params, np.float32(1e-6)))
            self.t = 100 * dt
        cp.cuda.Device().synchronize()


lorenz96 = '''
dydt[0] = (y[1] - y[3]) * y[4] - y[0] + p[0];
dydt[1] = (y[2] - y[4]) * y[0] - y[1] + p[0];
dydt[2] = (y[3] - y[0]) * y[1] - y[2] + p[0];
dydt[3] = (y[4] - y[1]) * y[2] - y[3] + p[0];
dydt[4] = (y[0] - y[2]) * y[3] - y[4] + p[0];
'''

def lorenz96_example():

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
    forward_euler_solver = ForwardEulerSolver(lorenz96, n_odes, n_vars, n_params)
    batched_ode_solver = BatchedODESolver(forward_euler_solver, 0, y, p)

    # Launch the Kernel for a single step
    batched_ode_solver.launch(0.01)

    # Output
    print('t=0.01 (first two states):')
    print(y[:2])

    # One more time for good measure
    batched_ode_solver.launch(0.01)
    print('t=0.02 (first two states):')
    print(y[:2])


def strong_scaling_forward_euler_singlestep_kernel():

    n_odes_vec = 1000000 * np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n_vars = 5
    n_params = 1
    time_vec = []

    for n_odes in n_odes_vec:

        # Generate initial condition data
        y = cp.random.uniform(-5, 5, (n_odes, n_vars), dtype=cp.float32) # type: ignore
        p = cp.random.uniform( 1, 5, (n_odes, n_params), dtype=cp.float32) # type: ignore

        start_time = time.perf_counter()

        # Batched ODE Solve
        forward_euler_solver = ForwardEulerSolver(lorenz96, n_odes, n_vars, n_params)
        batched_ode_solver = BatchedODESolver(forward_euler_solver, 0, y, p)
        for _ in range(100):
            batched_ode_solver.launch(dt=0.01, solver='rk23_singlestep')

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_vec.append(elapsed_time)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.semilogx(n_odes_vec, time_vec, '-or')
    ax.set_xlabel('Number of ODEs')
    ax.set_ylabel('Time elapsed (seconds)')
    ax.set_title('Strong Scaling, ForwardEulerSolverSinglestep (but launching the kernel for many steps)')
    fig.savefig('figs/strong_scaling_forward_euler_kernel_singlestep.png')



def strong_scaling_forward_euler_multistep_kernel():

    n_odes_vec = 1000000 * np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n_vars = 5
    n_params = 1
    time_vec = []

    for n_odes in n_odes_vec:

        # Generate initial condition data
        y = cp.random.uniform(-5, 5, (n_odes, n_vars), dtype=cp.float32) # type: ignore
        p = cp.random.uniform( 1, 5, (n_odes, n_params), dtype=cp.float32) # type: ignore

        start_time = time.perf_counter()

        # Batched ODE Solve
        forward_euler_solver = ForwardEulerSolverMultistep(lorenz96, n_odes, n_vars, n_params)
        batched_ode_solver = BatchedODESolver(forward_euler_solver, 0, y, p)
        batched_ode_solver.launch(dt=0.01, solver='rk23_multistep')

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_vec.append(elapsed_time)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.semilogx(n_odes_vec, time_vec, '-or')
    ax.set_xlabel('Number of ODEs')
    ax.set_ylabel('Time elapsed (seconds)')
    ax.set_title('Strong Scaling, ForwardEulerSolverMultistep (launching the kernel once, doing time stepping in the kernel)')
    fig.savefig('figs/strong_scaling_forward_euler_kernel_multistep.png')

    


def strong_scaling_rk23_kernel():

    n_odes_vec = 1000000 * np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    n_vars = 5
    n_params = 1
    time_vec = []

    for n_odes in n_odes_vec:

        # Generate initial condition data
        y = cp.random.uniform(-5, 5, (n_odes, n_vars), dtype=cp.float32) # type: ignore
        p = cp.random.uniform( 1, 5, (n_odes, n_params), dtype=cp.float32) # type: ignore

        start_time = time.perf_counter()

        # Batched ODE Solve
        rk23_solver = RK23SolverVarstep(lorenz96, n_odes, n_vars, n_params)
        batched_ode_solver = BatchedODESolver(rk23_solver, 0, y, p)
        batched_ode_solver.launch(dt=0.01, solver='rk23_multistep')

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        time_vec.append(elapsed_time)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.semilogx(n_odes_vec, time_vec, '-or')
    ax.set_xlabel('Number of ODEs')
    ax.set_ylabel('Time elapsed (seconds)')
    ax.set_title('Strong Scaling, RK23Multistep')
    fig.savefig('figs/strong_scaling_rk23_kernel.png')

    





if __name__ == '__main__':
    # lorenz96_example()
    # strong_scaling_forward_euler_singlestep_kernel()
    # strong_scaling_forward_euler_multistep_kernel()
    strong_scaling_rk23_kernel()
