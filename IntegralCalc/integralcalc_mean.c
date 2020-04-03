//Definite integral calculator
//Monte Carlo Sample-Mean method
//Also works with improper integrals most of the time

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
#include <limits.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"

//Third seed
static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint32_t h, l;
    __asm__ __volatile__("rdtsc" : "=a" (l), "=d" (h));
        val = ((uint64_t)l) | (((uint64_t)h) << 32);
        return val;
}

void initializeSeeds(cl_uint * seed1, cl_uint * seed2, cl_uint * seed3, cl_uint * seed4){
	*seed1 = time(0) & 134217727;
	*seed2 = getpid() & 131071;
	*seed3 = clock() & 131071;
	*seed4 = rdtsc() & 134217727;
}

//Method to replace the function in montecarlo.ocl
int replaceFunction(char * f){
	FILE * code, * temp;
	char str[MAX];
	int linectr = 0;

	char newln[MAX] = "\t\t";
	strcat(newln, f);
	strcat(newln, ";\n");

	code = fopen("montecarlo.ocl", "r");
	temp = fopen("temp.txt", "w");

	while(!feof(code)){
		strcpy(str, "\0");
		fgets(str, MAX, code);
		if(!feof(code)){
			linectr++;
			if(linectr == 4){
				fprintf(temp, "%s", newln);
			}
			else fprintf(temp, "%s", str);
		}
	}
	fclose(code);
	fclose(temp);
	remove("montecarlo.ocl");
	rename("temp.txt", "montecarlo.ocl");
	return 1;
}

//Setting up the kernel to compute function values in random samples
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que,
	cl_mem d_sum, float leftLimit, float rightLimit, cl_uint seed1, cl_uint seed2, cl_uint seed3, cl_uint seed4, cl_int nocts){

	printf("nocts: %d\n", nocts);
	const size_t gws[] = { nocts };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_sum), &d_sum);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimit), &rightLimit);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed1), &seed1);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed2), &seed2);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed3), &seed3);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed4), &seed4);
	ocl_check(err, "set montecarlo arg ", i-1);

	err = clEnqueueNDRangeKernel(que, montecarlo_k, 1, NULL, gws, NULL,
		0, NULL, &montecarlo_evt);
	ocl_check(err, "enqueue montecarlo");

	return montecarlo_evt;	
}

//Setting up the kernel to reduce the sum buffer
cl_event sum_reduction(cl_kernel reduce4_k, cl_command_queue que,
	cl_mem d_out, cl_mem d_in, cl_int nquarts,
	cl_int lws_, cl_int nwg,
	cl_event init_evt){

	printf("gws: %d, lws: %d\n", nwg*lws_, lws_);
	const size_t gws[] = { nwg*lws_ };
	const size_t lws[] = { lws_ };

	cl_event reduce4_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reduce4_k, i++, sizeof(d_out), &d_out);
	ocl_check(err, "set reduce4 arg", i-1);
	err = clSetKernelArg(reduce4_k, i++, sizeof(d_in), &d_in);
	ocl_check(err, "set reduce4 arg", i-1);
	err = clSetKernelArg(reduce4_k, i++, sizeof(float)*lws[0], NULL);
	ocl_check(err, "set reduce4 arg", i-1);
	err = clSetKernelArg(reduce4_k, i++, sizeof(nquarts), &nquarts);
	ocl_check(err, "set reduce4 arg", i-1);

	err = clEnqueueNDRangeKernel(que, reduce4_k, 1,
		NULL, gws, lws,
		1, &init_evt, &reduce4_evt);

	ocl_check(err, "enqueue reduce4_lmem");

	err = clFinish(que);

	ocl_check(err, "finish que");


	return reduce4_evt;
}

//The area under the curve is approximately the mean of the function values in the samples times the interval size
double computeIntegral(const double sumValue, const double intervalSize, const int npoints){
 	return intervalSize * (sumValue / npoints);
}

void check_accuracy(const double calculatedValue, const double expectedValue){
	printf("Expected value is %g, calculated value is %g\n", expectedValue, calculatedValue);
	double absErr = expectedValue - calculatedValue;
	if(absErr < 0) absErr *= -1;
	printf("The absolute error is %g\n", absErr);
}

void printArray(const float * v1, int npoints){
	for(int i=0; i<npoints; ++i){
		printf("(%f, %f) ", v1[2*i], v1[2*i+1]);
	}
	printf("\n");
}

int main(int argc, char* argv[]){

	int npoints = 128; 
	double leftLimit = 0.0;
	double rightLimit = 1.0;
	if(argc <= 4){
		printf("This program computes the area in an interval\nunder the curve of a function.\n");
		fprintf(stderr, "Use: %s f(x) leftLimit rightLimit npoints [expectedValue]\n", argv[0]);
		printf("Examples: %s 'sqrt(1-x*x)' 0.0 1.0 1024 0.78539825\n", argv[0]);
		printf("%s 'log(1+x)/x' 0.0 1.0 4096 0.822465644\n", argv[0]);
		printf("%s 'exp(-x*x)' 0.5 6.3 4096 0.424946\n", argv[0]);
		exit(0);
	}
	else{
		npoints = atoi(argv[4]);
			if(npoints & 7){
				npoints = round_mul_up(npoints, 8);
				fprintf(stderr, "npoints must be a multiple of 8. Rounding up to %d\n", npoints);
		}
		leftLimit = strtod(argv[2], NULL);
		rightLimit = strtod(argv[3], NULL);
		if(leftLimit > rightLimit){
			fprintf(stderr, "The left limit is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimit;
			leftLimit = rightLimit;
			rightLimit = temp;
		}
	}
	double intervalSize = rightLimit - leftLimit;
	replaceFunction(argv[1]);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("montecarlo.ocl", ctx, d);
	cl_int err;

	cl_kernel montecarlo_k = clCreateKernel(prog, "montecarlo_mean", &err);
	ocl_check(err, "create kernel montecarlo");
	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	//seeds for the edited MWC64X
	cl_uint seed1, seed2, seed3, seed4;
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	printf("Seeds: %d, %d, %d, %d\n", seed1, seed2, seed3, seed4);

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	if (npoints/8 < lws){
		nwg = 1;
		lws = npoints/8;
	}
	else nwg = round_mul_up(npoints/8, lws)/lws;
	//if (nwg<4 && nwg != 1) nwg=4;
	printf("lws: %ld, npoints/8: %d, nwg: %d\n", lws, npoints/8, nwg);
	size_t memsize = (npoints/2)*sizeof(float);
	const size_t nwg_mem = nwg*sizeof(float);


	cl_mem d_sum1, d_sum2;	//needed to save the function values and reduce them

	d_sum1 = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_sum1");
	d_sum2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		nwg_mem, NULL,
		&err);
	ocl_check(err, "create buffer d_sum2");


	cl_event montecarlo_evt, reduce_evt[2], read_evt;
	
	montecarlo_evt = montecarlo(montecarlo_k, que, d_sum1, leftLimit, rightLimit, seed1, seed2, seed3, seed4, npoints/8);

	// reducing the original data size to nwg elements
	reduce_evt[0] = sum_reduction(reduce4_k, que, d_sum2, d_sum1, npoints/8,
		lws, nwg, montecarlo_evt);
	// Wrap up the reduction
	if (nwg > 1) {
		reduce_evt[1] = sum_reduction(reduce4_k, que, d_sum2, d_sum2, round_mul_up(nwg,4)/4,
			lws, 1, reduce_evt[0]);
	} else {
		reduce_evt[1] = reduce_evt[0];
	}

	float sum_result;
	err = clEnqueueReadBuffer(que, d_sum2, CL_TRUE, 0, sizeof(sum_result), &sum_result,
		1, reduce_evt + 1, &read_evt);
	ocl_check(err, "read result");

	printf("The sum is %f\n\n", sum_result);

	double calculatedValue = computeIntegral(sum_result, intervalSize, npoints);
	printf("(%s, %s)∫%s dx = %g\n\n", argv[2], argv[3], argv[1], calculatedValue);
	if (argv[5]) check_accuracy(calculatedValue, strtod(argv[5], NULL));

	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	double montecarlo_bw_gbs = memsize/1.0e6/runtime_montecarlo_ms;
	double read_bw_gbs = sizeof(cl_float)/1.0e6/runtime_read_ms;

	printf("montecarlo : %d points (float, float) in %gms: %g GB/s\n",
		npoints, runtime_montecarlo_ms, montecarlo_bw_gbs);

	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[0]);
		const double pass_bw_gbs = (memsize+nwg_mem)/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			npoints, runtime_pass_ms, pass_bw_gbs,
			npoints/1.0e6/runtime_pass_ms);
	}
	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[1]);
		const double pass_bw_gbs = (nwg_mem+sizeof(float))/1.0e6/runtime_pass_ms;
		printf("reduce1 : %ld float in %gms: %g GB/s %g GE/s\n",
			(lws*nwg), runtime_pass_ms, pass_bw_gbs,
			(lws*nwg)/1.0e6/runtime_pass_ms);
	}
	const double runtime_reduction_ms = total_runtime_ms(reduce_evt[0], reduce_evt[1]);
	const double total_time_ms = total_runtime_ms(montecarlo_evt, read_evt);
	printf("reduce : %d float in %gms: %g GE/s\n",
		npoints, runtime_reduction_ms, npoints/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	
	//Leftovers to print the formatted information needed to do plots like runtime against gws
	//double absErr = strtod(argv[5], NULL) - calculatedValue;
	//if(absErr < 0) absErr *= -1;
	//printf("\nTotal:%d;%g;%g\n", npoints/8, total_time_ms, absErr);

	clReleaseMemObject(d_sum1);
	clReleaseMemObject(d_sum2);

	clReleaseKernel(montecarlo_k);
	clReleaseKernel(reduce4_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}