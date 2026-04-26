import cupy as cp

class ODESolver:

    def __init__(self, cuda_code: str, rhs_code: str, n_odes: int, n_vars: int, n_params: int):
        self.cuda_code = cuda_code.replace('$RHS_HERE$', rhs_code).replace('$N_ODES$', str(n_odes)).replace('$N_VARS$', str(n_vars)).replace('$N_PARAMS$', str(n_params))
        self.rhs_code = rhs_code
        self.n_vars = n_vars
        self.n_params = n_params
        self.ode_solver_raw_module = cp.RawModule(code=self.cuda_code)


class ForwardEulerSolver(ODESolver):

    def __init__(self, rhs_code: str, n_odes: int, n_vars: int, n_params: int):

        forward_euler_cuda_code_template = \
r'''
extern "C" {
                              
    // Contains RHS_HERE (the rhs code), N_ODES (number of ODEs), N_VARS (number of ODE variables), N_PARAMS (number of rhs parameters)
    // Dollar signs are where we replace the variable in the string before compiling

    __global__ void timestep_kernel(float t, float dt, float* y_all, float* params) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (i < $N_ODES$) {

            int ode_offset = i * $N_VARS$;
            int param_offset = i * $N_PARAMS$;
            float y[$N_VARS$];
            float p[$N_PARAMS$];

            // 1. Load current state and parameters into registers
            for(int d=0; d < $N_VARS$; d++) y[d] = y_all[ode_offset + d];
            for(int j=0; j < $N_PARAMS$; j++) p[j] = params[param_offset + j];

            // 2. Compute RHS
            float dydt[$N_VARS$];
            $RHS_HERE$

            // 3. Update and write back
            for(int d=0; d < $N_VARS$; d++) y_all[ode_offset + d] += dt * dydt[d];
        }
    }
}
'''

        super().__init__(forward_euler_cuda_code_template, rhs_code, n_odes, n_vars, n_params)

