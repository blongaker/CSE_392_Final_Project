import cupy as cp

class ODESolver:

    def __init__(self, cuda_code: str, n_vars: int, n_params: int):
        self.cuda_code = cuda_code.replace('$N_VARS$', str(n_vars)).replace('$N_PARAMS$', str(n_params))
        self.n_vars = n_vars
        self.n_params = n_params
        self.ode_solver_raw_kernel = cp.RawModule(code=self.cuda_code)


class ForwardEulerSolver(ODESolver):

    def __init__(self, n_vars: int, n_params: int):

        forward_euler_cuda_code_template = r'''
extern "C" {
                              
    // Contains N_SYSTEMS (number of ODEs), $N_VARS$ (number of ODE variables), $N_PARAMS$ (number of rhs parameters)

    // RHSFunc: (dydt_out, t, y_in, params)
    typedef void (*RHSFunc)(float*, const float, float*, const float*);

    __global__ void timestep_kernel(RHSFunc rhs, float t, float dt, float* y_all, float* params) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (i < n_systems) {
            int ode_offset = i * $N_VARS$;
            int param_offset = i * $N_PARAMS$;
            float y_local[$N_VARS$];
            float dydt_local[$N_VARS$];
            float params_local[$N_PARAMS$];

            // 1. Load current state and parameters into registers
            for(int d=0; d < $N_VARS$; d++) y_local[d] = y_all[ode_offset + d];
            for(int p=0; p < $N_PARAMS$; p++) params_local[p] = params[param_offset + p]

            // 2. Compute RHS
            rhs(dydt_local, t, y_local, params_local);

            // 3. Update and write back
            for(int d=0; d < $N_VARS$; d++) {
                y_all[ode_offset + d] = y_local[d] + dt * dydt_local[d];
            }
        }
    }
}
'''

        super().__init__(forward_euler_cuda_code_template, n_vars, n_params)
