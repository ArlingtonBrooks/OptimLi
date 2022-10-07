#include "OptimLi.hpp"
#include "OptimLi.cpp"

#include <iostream>

//Example parameter structure
struct paramspec {
	double a;
	double b;
};

//Example objective function
double F(double* x, void* params)
{
	paramspec* para = (paramspec*)params;
	double a = para->a;
	double b = para->b;

	double fv = -std::exp(-x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1]);

	return -std::exp(-x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1]);
}

//Example gradient
void dF(double* ret, double* x, void* params)
{
	paramspec* para = (paramspec*)params;
	double a = para->a;
	double b = para->b;

	ret[0] = -std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(a - 2.0*x[0]);
	ret[1] = -std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(b - 2.0*x[1]);
}

//Example hessian
void ddF(double* ret, double* x, void* params)
{
	paramspec* para = (paramspec*)params;
	double a = para->a;
	double b = para->b;

	ret[0] = std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(- a*a + 4.0*a*x[0] - 4.0*x[0]*x[0] + 2.0);
	ret[1] = -std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(a - 2.0*x[0])*(b - 2.0*x[1]);
	ret[2] = -std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(a - 2.0*x[0])*(b - 2.0*x[1]);
	ret[3] = std::exp(- x[0]*x[0] + a*x[0] - x[1]*x[1] + b*x[1])*(- b*b + 4.0*b*x[1] - 4.0*x[1]*x[1] + 2.0);
}

//An example implementation which performs an optimization of the given functions with each solver type.
int main(int argc, char** argv) 
{
	double inputs[2] = {3.0,3.0};
	paramspec apar {0.0,0.10};
	unsigned testval;
	printf("Running solver\n");
	int retcode = FMIN_Gradient(2,inputs,&apar,&F,&dF,NULL,0.0,1e-10,1400,&testval);
	printf("(grad)ret:%d Solved to %lg,%lg with fval=%lg   %u iters\n",retcode,inputs[0],inputs[1],F(inputs,&apar),testval);

	inputs[0] = 3.0;
	inputs[1] = 3.0;
	retcode = FMIN_NewtonSolver(2,inputs,&apar,&F,&dF,&ddF,0.0,1e-10,1400,&testval);
	printf("(newt1) Solved to %lg,%lg with fval=%lg   %u iters\n",inputs[0],inputs[1],F(inputs,&apar),testval);

	inputs[0] = 3.0;
	inputs[1] = 3.0;
	retcode = FMIN_DampNewtonSolver(2,inputs,&apar,&F,&dF,&ddF,0.0,1e-10,1400,0.1,&testval);
	printf("(newt2) %d Solved to %lg,%lg with fval=%lg   %u iters\n",retcode,inputs[0],inputs[1],F(inputs,&apar),testval);
	
	inputs[0] = 3.0;
	inputs[1] = 3.0;
	int RETTR = FMIN_SweepNewtonSolver(2,inputs,&apar,&F,&dF,&ddF,0.0,1e-10,1400,&testval);
	printf("(newt3) %d Solved to %lg,%lg with fval=%lg   %u iters\n",RETTR,inputs[0],inputs[1],F(inputs,&apar),testval);
	return 0;
}
