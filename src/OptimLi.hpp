#ifndef OPTIMLI_HPP__
#define OPTIMLI_HPP__

extern "C"
{
/*
dgesv_
    Compute solution to a linear system
    (external library)
*/
    void dgesv_(int *N, int *NRHS, double* A, int *LDA, int* IPIV, double* B, int *LDB, int *INFO);
}

/*
* Optimization algorithms
* NOTE:
*    All Newton solvers presented will attempt to take a Gradient Descent step if the Newton direction fails to yield a decrease in the objective function.
*/

//Optimality condition checker
bool CheckTol(int NumParams, double* grad, double* step, double AbsTol, double RelTol);

//A standard Newton solver
int FMIN_NewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount);
int FMIN_NewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter) 
{
	return FMIN_NewtonSolver(NumParams, params, args, feval, grad, hess, AbsTol, RelTol, MaxIter, 0);
}

//A Newton solver which attempts to select the best step size in the optimization direction with an approximate line search
int FMIN_SweepNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount);
int FMIN_SweepNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter) 
{
	return FMIN_SweepNewtonSolver(NumParams, params, args, feval, grad, hess, AbsTol, RelTol, MaxIter, 0);
}

//A Newton solver for which the maximum step in the search direction is defined by MaxStep
int FMIN_DampNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, double MaxStep, unsigned* IterCount);
int FMIN_DampNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, double MaxStep) 
{
	return FMIN_DampNewtonSolver(NumParams, params, args, feval, grad, hess, AbsTol, RelTol, MaxIter, MaxStep, 0);
}

//A Gradient Descent optimization algorithm with an approximate line search (EXPENSIVE)
int FMIN_Gradient(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount);
int FMIN_Gradient(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter) 
{
	return FMIN_Gradient(NumParams, params, args, feval, grad, hess, AbsTol, RelTol, MaxIter, 0);
}
#endif
