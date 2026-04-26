### CSE 392 Final Project: Batched ODE Solver.
The goal is to solve a system of ODEs of the form:
$$\begin{cases}
\dot{\mathbf{x}} = f(\mathbf{x}; \mathbf{p}) \\
\mathbf{x}(0) = \mathbf{x}_0,
\end{cases}$$
for multiple sets of initial conditions and parameters $\{\mathbf{x}_0, \mathbf{p}\}$. The steps for doing this are as follows:

1. Arrange the initial conditions into a single vector $\mathbf{x}_{\mathrm{big}}$, and build a modified right hand side $\mathbf{x}$
    - Use Cuml k-means to do initial clustering. Should be very easy (?)
2. In each cluster, pair together similar initial conditions and parameters and build modified right-hand-side accordingly.
    - Here, we need a wrapper function that takes the set of parameters and a right-hand side function, and returns a modified right-hand side that acts on batched states.
3. Send each batch to a single warp and solve.
    - Time stepping should be done in a raw kernel. Then we can make custom kernels with loops and whatnot.
    - Make a raw kernel for each time-stepping method.
    - Start with fixed time steps.
    - For non-uniform time-stepping, we would like groups of similar ODEs to go in the same warp (batches of 32 threads). This will be done if we have time.


___Saturday___: created a basic explicit (forward Euler) batched ODE solver. Each thread on the GPU handles a single ODE.

___Sunday:
- Started putting together slides for presentation
- Compare the now existing code with launching a time stepping loop inside each kernel. Which is faster? Profile and put these results in a table.
- Implement a variable time stepper (RK23?)
- How to parallelize this effectively?
    - Cluster initial conditions and parameters
    - Sort the ICs/params so that similar equations are together in blocks
    - Launch similar equations in a single warp so that time steps are the same.
