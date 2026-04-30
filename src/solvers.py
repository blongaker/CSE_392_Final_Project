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

    __device__ void rhs(float* dydt, float t, float* y, float* p) {
        $RHS_HERE$
    }

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
            rhs(dydt, t, y, p);

            // 3. Update and write back
            for(int d=0; d < $N_VARS$; d++) y_all[ode_offset + d] += dt * dydt[d];
        }
    }
}
'''

        super().__init__(forward_euler_cuda_code_template, rhs_code, n_odes, n_vars, n_params)


class ForwardEulerSolverMultistep(ODESolver):

    def __init__(self, rhs_code: str, n_odes: int, n_vars: int, n_params: int):

        forward_euler_cuda_code_template = \
r'''
extern "C" {
                              
    // Contains RHS_HERE (the rhs code), N_ODES (number of ODEs), N_VARS (number of ODE variables), N_PARAMS (number of rhs parameters)
    // Dollar signs are where we replace the variable in the string before compiling

    __device__ void rhs(float* dydt, float t, float* y, float* p) {
        $RHS_HERE$
    }

    __global__ void timestep_kernel(float t0, float dt, float tf, float* y_all, float* params) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (i < $N_ODES$) {

            int ode_offset = i * $N_VARS$;
            int param_offset = i * $N_PARAMS$;
            float y[$N_VARS$];
            float p[$N_PARAMS$];
            float dydt[$N_VARS$];

            // 1. Load current state and parameters into registers
            for(int d=0; d < $N_VARS$; d++) y[d] = y_all[ode_offset + d];
            for(int j=0; j < $N_PARAMS$; j++) p[j] = params[param_offset + j];

            // Loop!
            for(float t_curr = t0; t_curr < tf; t_curr += dt) {
                // Compute RHS
                rhs(dydt, t_curr, y, p);

                // Update current state
                for(int d=0; d < $N_VARS$; d++) y[d] += dt * dydt[d];
            }

            // 3. Write back
            for(int d=0; d < $N_VARS$; d++) y_all[ode_offset + d] = y[d];
        }
    }
}
'''

        super().__init__(forward_euler_cuda_code_template, rhs_code, n_odes, n_vars, n_params)



class RK23SolverVarstep(ODESolver):

    def __init__(self, rhs_code: str, n_odes: int, n_vars: int, n_params: int):

        forward_euler_cuda_code_template = \
r'''
extern "C" {
                              
    // Contains RHS_HERE (the rhs code), N_ODES (number of ODEs), N_VARS (number of ODE variables), N_PARAMS (number of rhs parameters)
    // Dollar signs are where we replace the variable in the string before compiling

    __device__ void rhs(float* dydt, float t, float* y, float* p) {
        $RHS_HERE$
    }

    __global__ void timestep_kernel(float t0, float dt, float tf, float* y_all, float* params, float tol) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        
        if (i < $N_ODES$) {

            int ode_offset = i * $N_VARS$;
            int param_offset = i * $N_PARAMS$;
            float y[$N_VARS$];
            float p[$N_PARAMS$];

            // Load current state and parameters into registers
            for(int d=0; d < $N_VARS$; d++) y[d] = y_all[ode_offset + d];
            for(int j=0; j < $N_PARAMS$; j++) p[j] = params[param_offset + j];

            // Registers we will need
            float dt_curr = dt;
            float t_curr = t0;
            float k1[$N_VARS$];
            float k2[$N_VARS$];
            float k3[$N_VARS$];
            float k4[$N_VARS$];
            float error_norm;
            float y_next[$N_VARS$];

            // Begin timestepping
            while (t_curr < tf) {

                // Compute RHS
                // Use the registers (y_next) we have already allocated for later
                rhs(k1, t_curr, y, p);
                for(int d=0; d < $N_VARS$; d++) y_next[d] = y[d] + 0.5  * dt_curr * k1[d];
                rhs(k2, t_curr + 0.5 * dt_curr, y_next, p);
                for(int d=0; d < $N_VARS$; d++) y_next[d] = y[d] + 0.75 * dt_curr * k2[d];
                rhs(k3, t_curr + 0.75 * dt_curr, y_next, p);

                // Update current state
                for(int d=0; d < $N_VARS$; d++) y_next[d] = y[d] + dt * (2*k1[d]+3*k2[d]+4*k3[d]) / 9;

                // Compute error
                rhs(k4, t_curr + dt_curr, y_next, p);
                error_norm = 0.0f;
                for(int d=0; d < $N_VARS$; d++)
                    error_norm += powf(dt_curr * (-5*k1[d]+6*k2[d]+8*k3[d]-9*k4[d]) / 72, 2);
                error_norm = powf(error_norm, 0.5);

                // Accept or reject solution based on error_norm
                // Introducing thread divergence here
                if (error_norm < tol) {
                    // Accept step
                    t_curr += dt_curr;
                    for(int d=0; d < $N_VARS$; d++) y[d] = y_next[d];
                }
                // Otherwise do nothing; reject step
                
                // Compute next time step
                dt_curr = min(0.9 * dt_curr * powf(tol / error_norm, 0.5), tf - t_curr);
            }

            // After time stepping is complete, update global state
            for(int d=0; d < $N_VARS$; d++) y_all[ode_offset + d] = y[d];

        }
    }
}
'''
        super().__init__(forward_euler_cuda_code_template, rhs_code, n_odes, n_vars, n_params)
