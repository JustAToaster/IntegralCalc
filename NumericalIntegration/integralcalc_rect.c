//Definite integral calculator with the rectangle rule

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"

//Method to replace the function in intcalc.ocl
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

//Setting up the kernel to compute the heights of each rectangle
cl_event intcalc(cl_kernel intcalc_k, cl_command_queue que,
	cl_mem d_v, float leftLimit, float base, int nrect){

	const size_t gws[] = { nrect/4 };

	cl_event intcalc_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_v), &d_v);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(base), &base);
	ocl_check(err, "set intcalc arg ", i-1);
	
	err = clEnqueueNDRangeKernel(que, intcalc_k, 1, NULL, gws, NULL,
		0, NULL, &intcalc_evt);
	ocl_check(err, "enqueue intcalc");

	return intcalc_evt;	
}

//Setting up the kernel to sum the function values
cl_event rect_reduction(cl_kernel reduce4_k, cl_command_queue que,
	cl_mem d_out, cl_mem d_in, cl_int nquarts,
	cl_int lws_, cl_int nwg,
	cl_event init_evt){

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

void check_accuracy(const double calculatedValue, const double expectedValue){
	printf("Expected value is %f, calculated value is %f\n", expectedValue, calculatedValue);
	double absErr = expectedValue - calculatedValue;
	if(absErr < 0) absErr *= -1;
	printf("The absolute error is %f\n", absErr);
}

int main(int argc, char* argv[]){

	int nrect = 128; 
	int lws = 16;
	int nwg = 1024;
	double leftLimit = 0.0;
	double rightLimit = 1.0;
	if(argc <= 4){
		printf("This program computes the area in an interval under the curve of a function with nrect rectangles.\n");
		fprintf(stderr, "Use: %s f(x) leftLimit rightLimit nrect [expectedValue]\n", argv[0]);
		printf("Examples: %s 'sqrt(1-x*x)' 0.0 1.0 1024 0.78539825\n", argv[0]);
		printf("%s 'log(1+x)/x' 0.0 1.0 4096 0.822465644\n", argv[0]);
		printf("%s 'exp(-x*x)' 0.5 6.3 4096 0.424946\n", argv[0]);
		exit(0);
	}
	else{
		nrect = atoi(argv[4]);
			if(nrect & 3){
				nrect = round_mul_up(nrect, 4);
				fprintf(stderr, "nrect must be a multiple of 4. Rounding up to %d\n", nrect);
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
	replaceFunction(argv[1]);

	float base = (rightLimit-leftLimit)/(nrect);
	size_t memsize = nrect*sizeof(float);
	const size_t nwg_mem = nwg*sizeof(float);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;
	cl_kernel intcalc_k;

	intcalc_k = clCreateKernel(prog, "intcalc_rect", &err);
	ocl_check(err, "create kernel intcalc");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	size_t gws_align_intcalc;
	err = clGetKernelWorkGroupInfo(intcalc_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(gws_align_intcalc), &gws_align_intcalc, NULL);
	ocl_check(err, "Preferred wg multiple for intcalc");
	int _gws = round_mul_up(nrect/4, gws_align_intcalc);
	printf("nrect/4: %d, _gws: %d\n", nrect/4, _gws);

	cl_mem d_v1, d_v2;

	d_v1 = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		nrect*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		nwg_mem, NULL,
		&err);
	ocl_check(err, "create buffer d_v2");

	cl_event intcalc_evt, reduce_evt[2], read_evt;
	
	intcalc_evt = intcalc(intcalc_k, que, d_v1, leftLimit, base, nrect);

	reduce_evt[0] = rect_reduction(reduce4_k, que, d_v2, d_v1, nrect/4,
		lws, nwg, intcalc_evt);
	if (nwg > 1) {
		reduce_evt[1] = rect_reduction(reduce4_k, que, d_v2, d_v2, nwg/4,
			lws, 1, reduce_evt[0]);
	} else {
		reduce_evt[1] = reduce_evt[0];
	}

	float sum_result;
	err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(sum_result), &sum_result,
		1, reduce_evt + 1, &read_evt);
	ocl_check(err, "read result");

	double calculatedValue = base * sum_result;

	printf("\n(%s, %s)∫%s dx = %g\n\n", argv[2], argv[3], argv[1], calculatedValue);
	if (argv[5]) check_accuracy(calculatedValue, strtod(argv[5], NULL));

	double runtime_intcalc_ms = runtime_ms(intcalc_evt);
	double runtime_read_ms = runtime_ms(read_evt);
	

	double intcalc_bw_gbs = memsize/1.0e6/runtime_intcalc_ms;
	double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("intcalc : %d float in %gms: %g GB/s\n",
		nrect, runtime_intcalc_ms, intcalc_bw_gbs);
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[0]);
		const double pass_bw_gbs = (memsize+nwg_mem)/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			nrect, runtime_pass_ms, pass_bw_gbs,
			nrect/1.0e6/runtime_pass_ms);
	}
	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[1]);
		const double pass_bw_gbs = (nwg_mem+sizeof(float))/1.0e6/runtime_pass_ms;
		printf("reduce1 : %d float in %gms: %g GB/s %g GE/s\n",
			(lws*nwg), runtime_pass_ms, pass_bw_gbs,
			(lws*nwg)/1.0e6/runtime_pass_ms);
	}
	const double runtime_reduction_ms = total_runtime_ms(reduce_evt[0], reduce_evt[1]);
	const double total_time_ms = total_runtime_ms(intcalc_evt, read_evt);
	printf("reduce : %d float in %gms: %g GE/s\n",
		nrect, runtime_reduction_ms, nrect/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);

	clReleaseMemObject(d_v1);
	clReleaseMemObject(d_v2);

	clReleaseKernel(intcalc_k);
	clReleaseKernel(reduce4_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}