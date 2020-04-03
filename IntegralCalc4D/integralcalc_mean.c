//Triple integral calculator
//Monte Carlo Method "Sample-Mean"
//Works with improper integrals most of the time (as long as it does not generate a number out of the domain)

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"

//Third seed
static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint h, l;
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

//This method replaces the function in intcalc.ocl
int replaceFunction(char * f){
	FILE * code, * temp;
	char str[MAX];
	int linectr = 0;

	char newln[MAX] = "\t\t";
	strcat(newln, f);
	strcat(newln, ";\n");

	code = fopen("intcalc.ocl", "r");
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
	remove("intcalc.ocl");
	rename("temp.txt", "intcalc.ocl");
	return 1;
}

//Setting up the kernel to generate random points on the integration grid and compute the function at those points
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que,
	cl_mem d_sum, float leftLimitX, float rightLimitX,
	float leftLimitY, float rightLimitY, float leftLimitZ, float rightLimitZ, 
	cl_uint seed1, cl_uint seed2, cl_uint seed3, cl_uint seed4, int npoints){

	const size_t gws[] = { npoints/8 };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_sum), &d_sum);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimitX), &leftLimitX);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimitX), &rightLimitX);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimitY), &leftLimitY);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimitY), &rightLimitY);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimitZ), &leftLimitZ);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimitZ), &rightLimitZ);
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

//Setting up the kernel to sum the function values, so that we can get an estimate for the integral mean
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

//The 4D volume beneath the curve is approximately the mean of the function values at the random points * the 3D volume of the region of integration
double computeIntegral(const double sumValue, const double cuboidVolume, const int npoints){
 	return cuboidVolume * (sumValue / npoints);
}

void check_accuracy(const double calculatedValue, const double expectedValue){
	printf("Expected value is %f, calculated value is %f\n", expectedValue, calculatedValue);
	double absErr = expectedValue - calculatedValue;
	if(absErr < 0) absErr *= -1;
	printf("The absolute error is %f\n", absErr);
}

int main(int argc, char* argv[]){

	int npoints = 128; 
	double leftLimitX = 0.0;
	double rightLimitX = 1.0;
	double leftLimitY = 0.0;
	double rightLimitY = 1.0;
	double leftLimitZ = 0.0;
	double rightLimitZ = 1.0;
	if(argc <= 4){
		printf("This program computes the 4D volume enclosed by a cuboid\nunder the curve of a three-variable function\n");
		fprintf(stderr, "Use: %s f(x, y, z) leftLimitX rightLimitX leftLimitY rightLimitY leftLimitZ rightLimitZ npoints [expectedValue]\n", argv[0]);
		printf("Examples: %s 'x*x + y*y + z*z' 0.0 1.0 0.0 1.0 0.0 1.0 1024 1\n", argv[0]);
		exit(0);
	}
	else{
		npoints = atoi(argv[8]);
			if(npoints & 7){
				npoints = round_mul_up(npoints, 8);
				fprintf(stderr, "npoints must be a multiple of 8. Rounding up to %d\n", npoints);
		}
		leftLimitX = strtod(argv[2], NULL);
		rightLimitX = strtod(argv[3], NULL);
		if(leftLimitX > rightLimitX){
			fprintf(stderr, "The left limit (x) is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimitX;
			leftLimitX = rightLimitX;
			rightLimitX = temp;
		}
		leftLimitY = strtod(argv[4], NULL);
		rightLimitY = strtod(argv[5], NULL);
		if(leftLimitY > rightLimitY){
			fprintf(stderr, "The left limit (y) is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimitY;
			leftLimitY = rightLimitY;
			rightLimitY = temp;
		}
		leftLimitZ = strtod(argv[6], NULL);
		rightLimitZ = strtod(argv[7], NULL);
		if(leftLimitZ > rightLimitZ){
			fprintf(stderr, "The left limit (z) is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimitZ;
			leftLimitZ = rightLimitZ;
			rightLimitZ = temp;
		}

	}
	double cuboidVolume = (rightLimitX-leftLimitX)*(rightLimitY-leftLimitY)*(rightLimitZ-leftLimitZ);
	replaceFunction(argv[1]);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;

	cl_kernel montecarlo_k = clCreateKernel(prog, "montecarlo_mean", &err);
	ocl_check(err, "create kernel montecarlo");
	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	cl_uint seed1, seed2, seed3, seed4;
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	printf("Seeds: %d, %d, %d, %d\n", seed1, seed2, seed3, seed4);

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	if (npoints/8 < lws){
		nwg = 1;
	}
	else nwg = round_mul_up(npoints/8, lws)/lws;
	if (nwg<4 && nwg != 1) nwg=4;
	if(nwg != 1) nwg = round_mul_up(nwg, 4);	//Number of work-groups must be a multiple of 4 as well because of the second reduction phase
	printf("lws: %ld, npoints/8: %d, nwg: %d\n", lws, npoints/8, nwg);
	size_t memsize = (npoints/2)*sizeof(float);
	const size_t nwg_mem = nwg*sizeof(float);


	cl_mem d_sum1, d_sum2;

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
	
	montecarlo_evt = montecarlo(montecarlo_k, que, d_sum1, leftLimitX, rightLimitX, 
		leftLimitY, rightLimitY, leftLimitZ, rightLimitZ, seed1, seed2, seed3, seed4, npoints);

	// reduce datasize to nwg elements
	reduce_evt[0] = sum_reduction(reduce4_k, que, d_sum2, d_sum1, npoints/8,
		lws, nwg, montecarlo_evt);
	// conclude reduction
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

	clWaitForEvents(1, &read_evt);

	printf("The sum is %f\n\n", sum_result);

	double calculatedValue = computeIntegral(sum_result, cuboidVolume, npoints);
	printf("(%s, %s)∫(%s, %s)∫(%s, %s)∫[%s] dxdydz = %g\n\n", argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[1], calculatedValue);
	if (argv[9]) check_accuracy(calculatedValue, strtod(argv[9], NULL));

	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	double montecarlo_bw_gbs = memsize/1.0e6/runtime_montecarlo_ms;
	double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("montecarlo : %d points (float, float, float) in %gms: %g GB/s\n",
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
	//double absErr = strtod(argv[7], NULL) - calculatedValue;
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
