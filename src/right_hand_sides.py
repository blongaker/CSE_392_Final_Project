import cupy as cp

rhs_module = cp.RawModule(code=r'''
extern "C" {

    __device__ void predator_prey(float* res, const float* y, float t, const float* p) {
        // y[0] = Prey, y[1] = Predator
        // p[0]=alpha, p[1]=beta, p[2]=delta, p[3]=gamma
        res[0] = p[0] * y[0] - p[1] * y[0] * y[1];
        res[1] = p[2] * y[0] - p[3] * y[1] * y[1];
    }

    __device__ RHSFunc ptr_predator_prey = predator_prey;
}
''')