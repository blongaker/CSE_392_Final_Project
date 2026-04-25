import cupy as cp


solvers_module = cp.RawModule(code=r'''
extern "C" {
                              

    // RHSFunc: (dydt_out, t, y_in, params)
    typedef void (*RHSFunc)(float*, const float, float*, const float*);

    __global__ void forward_euler_kernel(RHSFunc rhs, float t, float dt, float* y_all, const float* params, int n_systems, int n_vars, int n_params) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (i < n_systems) {
            int ode_offset = i * ode_size;
            int param_offset = i * n_params;
            float y_local[ode_size];
            float dydt_local[ode_size];
            float params_local[n_params];

            // 1. Load current state and parameters into registers
            for(int d=0; d < ode_size; d++) y_local[d] = y_all[ode_offset + d];
            for(int p=0; p < n_params; p++) params_local[p] = params[param_offset + p]

            // 2. Compute RHS
            rhs(dydt_local, y_local, t, params_local);

            // 3. Update and write back
            for(int d=0; d < DIM; d++) {
                y_all[offset + d] = y_local[d] + dt * dydt_local[d];
            }
        }
    }
}
''')


