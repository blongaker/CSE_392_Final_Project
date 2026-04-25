import cupy as cp

class DifferentialEquation:

    """
    Takes in C code that defines an instance of the right-hand side of the ODE.
    - dydt is the output
    - t is the current time
    - y is the state (array)
    - p is the parameters (array)
    """
    def __init__(self, cuda_code: str, n_vars: int, n_params: int):

        cuda_code_rhs_template = \
        r'''
        extern "C" {
            __device__ void rhs(float* dydt, const float t, const float* y, const float* p) {
                $RHS_HERE$
            }
            __device__ RHSFunc ptr_rhs = rhs;
        }
        '''
        self.cuda_code = cuda_code_rhs_template.replace('$RHS_HERE$', cuda_code)
        self.rhs_module = cp.RawModule(code=self.cuda_code) # Compiles the code on the fly
        self.n_vars = n_vars
        self.n_params = n_params


# Example
if __name__ == '__main__':
    lotka_volterra = DifferentialEquation(
        cuda_code = r'''
        dydt[0] =  p[0] * y[0] - p[1] * y[0] * y[1];
        dydt[1] = -p[3] * y[1] + p[3] * y[0] * y[1];
        ''',
        n_vars = 2, n_params = 4
    )
    print(lotka_volterra.cuda_code)
    print(lotka_volterra.n_vars)
    print(lotka_volterra.n_params)
    