//Double integral calculator with the cuboid rule
//NOTE: won't work with improper integrals

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"

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
			if(linectr == 4 || linectr == 9){
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

//Setting up the kernel to get the samples from the function, which represent the height of each cuboid
cl_event intcalc(cl_kernel intcalc_k, cl_command_queue que,
	cl_mem d_funcValues, float leftLimitX, float leftLimitY,
	float offX, float offY, int nrectsPerDim){

	const size_t gws[] = { nrectsPerDim/2, nrectsPerDim/2 };

	cl_event intcalc_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_funcValues), &d_funcValues);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimitX), &leftLimitX);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimitY), &leftLimitY);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(offX), &offX);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(offY), &offY);
	ocl_check(err, "set intcalc arg ", i-1);

	//Two dimensional grid!
	err = clEnqueueNDRangeKernel(que, intcalc_k, 2, NULL, gws, NULL,
		0, NULL, &intcalc_evt);
	ocl_check(err, "enqueue intcalc");

	return intcalc_evt;	
}

//Setting up the kernel to sum the values, so that we can compute the total volume
cl_event vol_reduction(cl_kernel reduce4_k, cl_command_queue que,
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

	int nrectsPerDim = 128; 
	double leftLimitX = 0.0;
	double rightLimitX = 1.0;
	double leftLimitY = 0.0;
	double rightLimitY = 1.0;
	if(argc <= 6){
		printf("This program computes the volume enclosed by a rectangle\nunder the curve of a two-variable function\n");
		fprintf(stderr, "Use: %s f(x, y) leftLimitX rightLimitX leftLimitY rightLimitY nrectsPerDim [expectedValue]\n", argv[0]);
		printf("Examples: %s 'x*x + y*y' 0.0 1.0 0.0 1.0 1024 0.666666666\n", argv[0]);
		printf("%s 'x*x + 4*y' 11 14 7 10 256 1719\n", argv[0]);
		printf("%s 'x*sin(y)' 0 2 0 1.570796 2048 2\n", argv[0]);	
		exit(0);
	}
	else{
		nrectsPerDim = atoi(argv[6]);
			if(nrectsPerDim & 1){
				nrectsPerDim = round_mul_up(nrectsPerDim, 2);
				fprintf(stderr, "The number of samples per dimension must be a multiple of 2. Rounding up to %d\n", nrectsPerDim);
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

	}
	replaceFunction(argv[1]);

	float offX = (rightLimitX-leftLimitX)/(nrectsPerDim);
	float offY = (rightLimitY-leftLimitY)/(nrectsPerDim);

	int nvolumes = nrectsPerDim*nrectsPerDim;

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;
	cl_kernel intcalc_k;

	intcalc_k = clCreateKernel(prog, "intcalc_cuboids", &err);
	ocl_check(err, "create kernel intcalc");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, 
		sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	if (nvolumes/4 < lws){
		nwg = 1;
	}
	else nwg = round_mul_up(nvolumes/4, lws)/lws;
	if (nwg<4 && nwg != 1) nwg=4;
	if (nwg != 1) nwg = round_mul_up(nwg, 4);	//Number of work-groups must be a multiple of 4 as well because of the second reduction phase
	printf("lws: %ld, nvolumes/4: %d, nwg: %d\n", lws, nvolumes/4, nwg);

	size_t memsize = nvolumes*sizeof(float);
	const size_t nwg_mem = nwg*sizeof(float);

	cl_mem d_funcValues1, d_funcValues2;

	d_funcValues1 = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_funcValues1");
	d_funcValues2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		nwg_mem, NULL,
		&err);
	ocl_check(err, "create buffer d_funcValues2");

	cl_event intcalc_evt, reduce_evt[2], read_evt;
	intcalc_evt = intcalc(intcalc_k, que, d_funcValues1, leftLimitX, leftLimitY, offX, offY, nrectsPerDim);
	reduce_evt[0] = vol_reduction(reduce4_k, que, d_funcValues2, d_funcValues1, nvolumes/4,
		lws, nwg, intcalc_evt);
	// concludo
	if (nwg > 1) {
		reduce_evt[1] = vol_reduction(reduce4_k, que, d_funcValues2, d_funcValues2, nwg/4,
			lws, 1, reduce_evt[0]);
	} else {
		reduce_evt[1] = reduce_evt[0];
	}

	float sum_result;
	err = clEnqueueReadBuffer(que, d_funcValues2, CL_TRUE, 0, sizeof(sum_result), &sum_result,
		1, reduce_evt + 1, &read_evt);
	ocl_check(err, "read result");
	
	double calculatedValue = offX*offY*sum_result;
	printf("\n(%s, %s)∫(%s, %s)∫[%s] dxdy = %f\n\n", argv[2], argv[3], argv[4], argv[5], argv[1], calculatedValue);
	if (argv[7]) check_accuracy(calculatedValue, strtod(argv[7], NULL));

	double runtime_intcalc_ms = runtime_ms(intcalc_evt);
	double runtime_read_ms = runtime_ms(read_evt);
	
	double intcalc_bw_gbs = memsize/1.0e6/runtime_intcalc_ms;
	double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("intcalc : %d function values (float) in %gms: %g GB/s\n",
		nvolumes, runtime_intcalc_ms, intcalc_bw_gbs);
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[0]);
		const double pass_bw_gbs = (memsize+nwg_mem)/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			nvolumes, runtime_pass_ms, pass_bw_gbs,
			nvolumes/1.0e6/runtime_pass_ms);
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
	const double total_time_ms = total_runtime_ms(intcalc_evt, read_evt);
	printf("reduce : %d float in %gms: %g GE/s\n",
		nvolumes, runtime_reduction_ms, nrectsPerDim/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//double absErr = strtod(argv[7], NULL) - calculatedValue;
	//if(absErr < 0) absErr *= -1;
	//printf("\nTotal:%d;%g;%g\n", nrectsPerDim*nrectsPerDim/4, total_time_ms, absErr);

	clReleaseMemObject(d_funcValues1);
	clReleaseMemObject(d_funcValues2);

	clReleaseKernel(intcalc_k);
	clReleaseKernel(reduce4_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}