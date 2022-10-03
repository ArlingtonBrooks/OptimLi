//TODO: move this to header;
#ifndef OPTIMLI_CPP__
#define OPTIMLI_CPP__
#include "OptimLi.hpp"
#include <math.h>
#include <iostream>
/*
CheckTol
	A tolerance checking function (using L1 norm)
	*NumParams:	Number of parameters in vectors
	*gradient:	Gradient vector
	*step:		Step vector
	*AbsTol:	Absolute tolerance to check
	*RelTol:	Relative tolerance to check
	
RETURNS:
	true:	Both gradient and step values are within tolerance
	false:	Tolerance condition not met (or both tolerance conditions are zero)
*/
bool CheckTol(int NumParams, double* grad, double* step, double AbsTol, double RelTol)
{
	if (AbsTol <= 0.0 && RelTol <= 0.0)
		return false;
	
	double ComputedRelTol = 0.0;
	double ComputedAbsTol = 0.0;
	for (int i = 0; i < NumParams; i++){
		ComputedRelTol += (grad == 0) ? 0.0 : grad[i]*grad[i];
		ComputedAbsTol += (step == 0) ? 0.0 : step[i]*step[i];
	}
	return (((std::sqrt(ComputedRelTol) < RelTol && grad != 0) || grad == 0) &&
		((std::sqrt(ComputedAbsTol) < AbsTol && step != 0) || step == 0));
}

/*
FMIN_NewtonSolver
	A pure undamped newton solver fora function with defined gradient and hessian.  Solves to either AbsTol or RelTol (whichever comes first) or MaxIter iterations.
	*NumParams: number of parameters being tuned
	*params:  the parameters being tuned (assumed to be populated with an initial guess)
	*args:	  arguments necessary to evaluate function/gradient/hessian (stored as void pointer)
	*fval:	  function value function pointer
	*grad:	  gradient function pointer
	*hess:	  hessian function pointer
	*AbsTol:  Absolute tolerance to stop iterating (set to 0 to prefer relative tolerance)
	*RelTol:  Relative tolerance to stop iterating (set to 0 to prefer absolute)
	*MaxIter: Maximum number of iterations to perform
	*IterCount: Running counter for number of iterations performed (optional)
NOTES:
	*If AbsTol and RelTol are both zero, solver will run until MaxIter iterations.
RETURNS:
	Returns a status code of one of the following:
	*0: Successfully hit either AbsTol or RelTol
	*1: MaxIter exceeded
	*2: Solution is not proceeding (step size is less than machine zero for both newton and gradient step)
	*3: Ambiguous input
*/
int FMIN_NewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount)
{
	//Check for valid input
	if (NumParams < 0 || params == NULL || feval == NULL || grad == NULL || hess == NULL)
		return 3;
	
	double fval;
	double gradval[NumParams];
	double stepval[NumParams];
	double hessval[NumParams*NumParams];
	unsigned i = 0;
	double OldParams[NumParams];

	//Optimization Loop
	LOOP:
		for (int j = 0; j < NumParams; j++)
			OldParams[j] = params[j];
		//Evaluate Function, gradient, and hessian
		fval = (feval)(params,args);
		(grad)(gradval,params,args);
		(hess)(hessval,params,args);

		//Setup LAPACK variables
		int IPIV[NumParams*NumParams];
		int N = NumParams;
		int NRHS = 1;
		int LDA = N;
		int LDB = N;
		int INFO;

		//Compute step
		for (int j = 0; j < NumParams; j++)
			stepval[j] = gradval[j];

		try {dgesv_(&N,&NRHS,hessval,&LDA,IPIV,stepval,&LDB,&INFO);}
		catch (...) {throw INFO;}

		double CHK = 0.0;
		for (int j = 0; j < NumParams; j++)
			CHK += stepval[j]*gradval[j];
		//Apply step
		for (int j = 0; j < NumParams; j++)
			if (CHK < 0.0)
				params[j] += stepval[j];
			else
				params[j] -= stepval[j];
		//Check step size
		bool HasStepped = false;
		for (int j = 0; j < NumParams; j++)
			if (params[j] != OldParams[j]) {
				HasStepped = true;
				break;
			}

		//Apply gradient step if step size too small
		if (!HasStepped) {
			for (int j = 0; j < NumParams; j++) {
				params[j] += -gradval[j];
				if (params[j] != OldParams[j])
					HasStepped = true;
			}
		}

		if (!HasStepped)
			return 2;

		if (CheckTol(NumParams,gradval,stepval,AbsTol,RelTol) && i > 0)
			return 0;

		i++;
		if (IterCount)
			*IterCount = i;
		if (i > MaxIter)
			return 1;
	goto LOOP;
}

/*
FMIN_SweepNewtonSolver
	A Newton solver which attempts to sweep in the direction of the Newton step to obtain a better solution than that which is obtained by a pure Newton step; This should allow for both an increased and decreased step size;
	*NumParams: number of parameters being tuned
	*params:  the parameters being tuned (assumed to be populated with an initial guess)
	*args:	  arguments necessary to evaluate function/gradient/hessian (stored as void pointer)
	*fval:	  function value function pointer
	*grad:	  gradient function pointer
	*hess:	  hessian function pointer
	*AbsTol:  Absolute tolerance to stop iterating (set to 0 to prefer relative tolerance)
	*RelTol:  Relative tolerance to stop iterating (set to 0 to prefer absolute)
	*MaxIter:   Maximum number of iterations to perform
	*IterCount: Running counter for number of iterations performed (optional)
NOTES:
	*If AbsTol and RelTol are both zero, solver will run until MaxIter iterations.
RETURNS:
	Returns a status code of one of the following:
	*0: Successfully hit either AbsTol or RelTol
	*1: MaxIter exceeded
	*2: Solution is not proceeding (step size is less than machine zero for both newton and gradient step)
	*3: Ambiguous input
*/
int FMIN_SweepNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount)
{
	//Check for valid input
	if (NumParams < 0 || params == NULL || feval == NULL || grad == NULL || hess == NULL)
		return 3;

	//Determine whether to compute tolerance
	bool ComputeTol = true;
	if (AbsTol == 0.0 && RelTol == 0)
		ComputeTol = false;
	
	double fval;
	double gradval[NumParams];
	double stepval[NumParams];
	double hessval[NumParams*NumParams];
	unsigned i = 0;
	double OldParams[NumParams];

	//Optimization Loop
	LOOP:
		for (int j = 0; j < NumParams; j++)
			OldParams[j] = params[j];
		//Evaluate Function, gradient, and hessian
		fval = (feval)(params,args);
		(grad)(gradval,params,args);
		(hess)(hessval,params,args);

		//Setup LAPACK variables
		int IPIV[NumParams*NumParams];
		int N = NumParams;
		int NRHS = 1;
		int LDA = N;
		int LDB = N;
		int INFO;

		//Compute step
		for (int j = 0; j < NumParams; j++)
			stepval[j] = gradval[j];

		try {dgesv_(&N,&NRHS,hessval,&LDA,IPIV,stepval,&LDB,&INFO);}
		catch (...) {throw INFO;}

		//Step Restriction
		bool FirstStep = true;
		double StepSize = 0.0;
		for (int j = 0; j < NumParams; j++)
			StepSize += stepval[j]*stepval[j];
		StepSize = std::sqrt(StepSize);

		double CHK = 0.0;
		for (int j = 0; j < NumParams; j++)
			CHK += stepval[j]*gradval[j];

		double direction = (CHK < 0.0) ? 1.0 : -1.0;
		bool Increasing = false;
		double factor = 1.0;
		double BestFactor = 0.0;
		double BestFval = fval;
		double fnewval1;
		double fnewval2;
		//Step expansion
		for (int k = 0; k < 64; k++) { //this should always be decreasing;
			fnewval1 = (feval)(params,args);
			for (int j = 0; j < NumParams; j++)
				params[j] = OldParams[j] + direction*factor*stepval[j];
			fnewval2 = (feval)(params,args);
			if (isnan(fnewval2) || isinf(fnewval2)) {
				for (int j = 0; j < NumParams; j++)
					params[j] = OldParams[j];
				Increasing = false;
				factor = factor * 0.5;
				continue;
			}
			if (k == 0) { //on first iteration
				if (fnewval2 < BestFval) { //accept step if better than nothing
					Increasing = true;
					BestFactor = factor;
					BestFval = fnewval2;
					factor = factor * 1.5;
				} else { //Refine if new step is not better;
					Increasing = false;
					factor = factor * 0.75;
				}
				continue;
			}

			if (fnewval2 <= BestFval && Increasing) { //New step is good
				BestFactor = factor;
				BestFval = fnewval2;
				factor = factor * 1.25;
			} else if (fnewval2 <= BestFval && !Increasing) { //New step is good
				BestFactor = factor;
				BestFval = fnewval2;
				factor = factor * 0.75;
			} else if (fnewval2 >= fnewval1) { //Refine if new step worse
				if (Increasing)
					factor = factor*0.9;
				else
					factor = factor*1.1;
			} else if (fnewval2 < fnewval1) { //Refine if new step worse
				if (Increasing)
					factor = factor * 1.1;
				else
					factor = factor * 0.9;
			}
		}
		//printf("%lg,%lg,%lg,%lg\n",factor,BestFactor,fnewval1,fnewval2);
		if (isnan(fnewval1) || isnan(fnewval2) || isinf(fnewval1) || isinf(fnewval2))
			return 2;
		//If unable to find a factor, take a gradient step
		if (BestFactor == 0.0) {
			FMIN_Gradient(NumParams,params,args,feval,grad,NULL,0.0,0.0,0);
		} else {
		//Apply step
			for (int j = 0; j < NumParams; j++)
				params[j] = OldParams[j] + direction*BestFactor*stepval[j];
		}

		//Check step size
		bool HasStepped = false;
		for (int j = 0; j < NumParams; j++)
			if (params[j] != OldParams[j]) {
				HasStepped = true;
				break;
			}

		//Apply gradient step if step size too small
		if (!HasStepped) {
			FMIN_Gradient(NumParams,params,args,feval,grad,NULL,0.0,0.0,0);
		}

		if (CheckTol(NumParams,gradval,stepval,AbsTol,RelTol) && i > 0)
			return 0;

		i++;
		if (IterCount)
			*IterCount = i;
		if (i > MaxIter) {
			return 1;
		}
	goto LOOP;
}

/*
FMIN_DampNewtonSolver
	The pure newton solver, however the step size is limited by MaxStep for use with difficult optimization functions;
	*NumParams: number of parameters being tuned
	*params:  the parameters being tuned (assumed to be populated with an initial guess)
	*args:	  arguments necessary to evaluate function/gradient/hessian (stored as void pointer)
	*fval:	  function value function pointer
	*grad:	  gradient function pointer
	*hess:	  hessian function pointer
	*AbsTol:  Absolute tolerance to stop iterating (set to 0 to prefer relative tolerance)
	*RelTol:  Relative tolerance to stop iterating (set to 0 to prefer absolute)
	*MaxIter: Maximum number of iterations to perform
	*MaxStep: Maximum step size (for damping)
	*IterCount: Running counter for number of iterations performed (optional)
NOTES:
	*If AbsTol and RelTol are both zero, solver will run until MaxIter iterations.
RETURNS:
	Returns a status code of one of the following:
	*0: Successfully hit either AbsTol or RelTol
	*1: MaxIter exceeded
	*2: Solution is not proceeding (step size is less than machine zero for both newton and gradient step)
	*3: Ambiguous input
*/
int FMIN_DampNewtonSolver(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, double MaxStep, unsigned* IterCount)
{
	//Check for valid input
	if (NumParams < 0 || params == NULL || feval == NULL || grad == NULL || hess == NULL)
		return 3;

	//Determine whether to compute tolerance
	bool ComputeTol = true;
	if (AbsTol == 0.0 && RelTol == 0)
		ComputeTol = false;
	
	double fval;
	double gradval[NumParams];
	double stepval[NumParams];
	double hessval[NumParams*NumParams];
	unsigned i = 0;
	double OldParams[NumParams];

	//Optimization Loop
	LOOP:
		for (int j = 0; j < NumParams; j++)
			OldParams[j] = params[j];
		//Evaluate Function, gradient, and hessian
		fval = (feval)(params,args);
		(grad)(gradval,params,args);
		(hess)(hessval,params,args);

		//Setup LAPACK variables
		int IPIV[NumParams*NumParams];
		int N = NumParams;
		int NRHS = 1;
		int LDA = N;
		int LDB = N;
		int INFO;

		//Compute step
		for (int j = 0; j < NumParams; j++)
			stepval[j] = gradval[j];

		try {dgesv_(&N,&NRHS,hessval,&LDA,IPIV,stepval,&LDB,&INFO);}
		catch (...) {throw INFO;}

		//Damp
		double StepSize = 0.0;
		for (int j = 0; j < NumParams; j++)
			StepSize += stepval[j]*stepval[j];
		StepSize = std::sqrt(StepSize);
		if (StepSize == 0)
			return 0;
		if (StepSize > MaxStep)
			for (int j = 0; j < NumParams; j++)
				stepval[j] = MaxStep*stepval[j]/StepSize;

		double CHK = 0.0;
		for (int j = 0; j < NumParams; j++)
			CHK += stepval[j]*gradval[j];
		//Apply step
		for (int j = 0; j < NumParams; j++)
			if (CHK < 0.0)
				params[j] += stepval[j];
			else
				params[j] -= stepval[j];
		//Check step size
		bool HasStepped = false;
		for (int j = 0; j < NumParams; j++)
			if (params[j] != OldParams[j]) {
				HasStepped = true;
				break;
			}
		//Apply gradient step if step size too small
		if (!HasStepped) {
			FMIN_Gradient(NumParams,params,args,feval,grad,NULL,0.0,0.0,1);
			for (int j = 0; j < NumParams; j++) {
				if (params[j] != OldParams[j])
					HasStepped = true;
			}
		}

		if (!HasStepped)
			return 2;

		if (CheckTol(NumParams,gradval,stepval,AbsTol,RelTol) && i > 0)
			return 0;

		i++;
		if (IterCount)
			*IterCount = i;
		if (i > MaxIter)
			return 1;
	goto LOOP;
}

/*
FMIN_Gradient
	A function which attempts to optimize in the gradient descent direction with a simple backtracking line search
	*NumParams: number of parameters being tuned
	*params:  the parameters being tuned (assumed to be populated with an initial guess)
	*args:	  arguments necessary to evaluate function/gradient/hessian (stored as void pointer)
	*fval:	  function value function pointer
	*grad:	  gradient function pointer
	*hess:	  (unused for gradient descent)
	*AbsTol:  Absolute tolerance to stop iterating (set to 0 to prefer relative tolerance)
	*RelTol:  Relative tolerance to stop iterating (set to 0 to prefer absolute)
	*MaxIter: Maximum number of iterations to perform
	*IterCount: Running counter for number of iterations performed (optional)
NOTES:
	*If AbsTol and RelTol are both zero, solver will run until MaxIter iterations.
RETURNS:
	Returns a status code of one of the following:
	*0: Successfully hit either AbsTol or RelTol
	*1: MaxIter exceeded
	*2: Solution is not proceeding (step size is less than machine zero)
	*3: Ambiguous input
*/
int FMIN_Gradient(int NumParams, double* params, void* args, double feval(double*,void*), void grad(double*,double*,void*), void hess(double*,double*,void*), double AbsTol, double RelTol, unsigned MaxIter, unsigned* IterCount)
{
	//Check for valid input
	if (NumParams < 0 || params == NULL || feval == NULL || grad == NULL)
		return 3;

	//Determine whether to compute tolerance
	bool ComputeTol = true;
	if (AbsTol == 0.0 && RelTol == 0.0)
		ComputeTol = false;
	
	double fval;
	double gradval[NumParams];
	double stepval[NumParams];
	unsigned i = 0;
	double OldParams[NumParams];
	double ParamSize = 0.0;
	for (int j = 0; j < NumParams; j++) {
		ParamSize += params[j]*params[j];
		OldParams[j] = params[j];
	}
	ParamSize = std::sqrt(ParamSize);
	double tol = 1.0;

	//Optimization Loop
	LOOP:
		//Evaluate Function, gradient, and hessian
		fval = (feval)(params,args);
		(grad)(gradval,params,args);

		//NAN check
		if (isnan(fval) || isinf(fval))
			return 2;
		//Compute gradient size
		double mag = 0.0;
		ParamSize = 0.0;
		for (int j = 0; j < NumParams; j++) {
			OldParams[j] = params[j];
			ParamSize += params[j]*params[j];
			mag += gradval[j]*gradval[j];
		}
		mag = sqrt(mag);
		ParamSize = sqrt(ParamSize);
		double StepSize = (tol < ParamSize) ? tol : ParamSize;
		//ParamSize-length step
		for (int j = 0; j < NumParams; j++) {
			stepval[j] = -StepSize*gradval[j]/mag;
		}
		//Search for correct step length
		int ITERS = 0;
		while (ITERS < 150) {
			for (int j = 0; j < NumParams; j++) {
				params[j] = OldParams[j] + stepval[j];
			}
			double newfval = (feval)(params,args);
			if (newfval < fval) {
				break;
			} else {
				StepSize = 0.8*StepSize;
				for (int j = 0; j < NumParams; j++) {
					stepval[j] = -StepSize*gradval[j]/mag;
				}
			}
			if (isnan(newfval) || isinf(newfval)) {
				StepSize = StepSize * 0.5;
				ITERS++; //FIXME: this is returning NANs in the actual code;
				continue;
			}
			ITERS++;
		}
		//printf("parameters:%lg,%lg,%lg,%lg\n",params[0],params[1],params[2],params[3],params[4]);
		if (ITERS == 1)
			tol = 1.1*tol;
		if (ITERS > 50)
			tol = 0.9*tol;
		if (ITERS == 100)
			return 2;

		//Check step size
		bool HasStepped = false;
		for (int j = 0; j < NumParams; j++)
			if (params[j] != OldParams[j]) {
				HasStepped = true;
				break;
			}

		if (!HasStepped) {
			return 2;
		}

		if (CheckTol(NumParams,gradval,stepval,AbsTol,RelTol) && i > 0)
			return 0;

		i++;
		if (IterCount)
			*IterCount = i;
		if (i > MaxIter)
			return 1;
	goto LOOP;
}

#endif
