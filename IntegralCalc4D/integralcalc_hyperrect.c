//Triple integral calculator with the hyperrectangle rule
//NOTE: might not work with improper integrals

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"

void initializeVal(float * h_null, int nvolumes){
	for(int i=0; i<nvolumes; i++){
		h_null[i]=0.0f;
	}
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

//Setting up the kernel to get the samples from the function, which represent the height of each hyperrectangle
cl_event intcalc(cl_kernel intcalc_k, cl_command_queue que, cl_mem d_funcValues, 
	float leftLimitX, float leftLimitY, float leftLimitZ,
	float offX, float offY, float offZ, int ncuboidsPerDim){

	const size_t gws[] = { ncuboidsPerDim, ncuboidsPerDim, ncuboidsPerDim };

	cl_event intcalc_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_funcValues), &d_funcValues);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimitX), &leftLimitX);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimitY), &leftLimitY);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(leftLimitZ), &leftLimitZ);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(offX), &offX);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(offY), &offY);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(offZ), &offZ);
	ocl_check(err, "set intcalc arg ", i-1);

	//Three dimensional grid!
	err = clEnqueueNDRangeKernel(que, intcalc_k, 3, NULL, gws, NULL,
		0, NULL, &intcalc_evt);
	ocl_check(err, "enqueue intcalc");

	return intcalc_evt;	
}

//Setting up the kernel to sum the values, so that we can compute the total 4D volume
cl_event vol_reduction(cl_kernel reduce4_k, cl_command_queue que,
	cl_mem d_out, cl_mem d_in, cl_int nquarts,
	cl_int lws_, cl_int nwg,
	cl_event init_evt){
	printf("lws: %d, nwg: %d, gws: %d\n", lws_, nwg, nwg*lws_);

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

	int ncuboidsPerDim = 128; 
	double leftLimitX = 0.0;
	double rightLimitX = 1.0;
	double leftLimitY = 0.0;
	double rightLimitY = 1.0;
	double leftLimitZ = 0.0;
	double rightLimitZ = 1.0;
	if(argc <= 8){
		printf("This program computes the 4D volume enclosed by a cuboid\nunder the curve of a three-variable function\n");
		fprintf(stderr, "Use: %s f(x, y, z) leftLimitX rightLimitX leftLimitY rightLimitY leftLimitZ rightLimitZ ncuboidsPerDim [expectedValue]\n", argv[0]);
		printf("Examples: %s 'x*x + y*y + z*z' 0.0 1.0 0.0 1.0 0.0 1.0 16 1\n", argv[0]);
		exit(0);
	}
	else{
		ncuboidsPerDim = atoi(argv[8]);
		leftLimitX = strtod(argv[2], NULL);
		rightLimitX = strtod(argv[3], NULL);
		if(ncuboidsPerDim & 1){
			ncuboidsPerDim = round_mul_up(ncuboidsPerDim, 2);
			fprintf(stderr, "The number of samples per dimension must be even. Rounding up to %d\n", ncuboidsPerDim);
		}
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
	replaceFunction(argv[1]);

	float offX = (rightLimitX-leftLimitX)/(ncuboidsPerDim);
	float offY = (rightLimitY-leftLimitY)/(ncuboidsPerDim);
	float offZ = (rightLimitZ-leftLimitZ)/(ncuboidsPerDim);

	int nvolumes = (ncuboidsPerDim)*(ncuboidsPerDim)*(ncuboidsPerDim);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;
	cl_kernel intcalc_k;

	intcalc_k = clCreateKernel(prog, "intcalc_hyperrect", &err);
	ocl_check(err, "create kernel intcalc");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	int reduce_volumes = round_mul_up(nvolumes, 4);
	if (reduce_volumes/4 < lws){
		nwg = 1;
	}
	else nwg = round_mul_up(reduce_volumes/4, lws)/lws;
	if (nwg<4 && nwg != 1) nwg=4;
	if (nwg != 1) nwg = round_mul_up(nwg, 4);	//Number of work-groups must be a multiple of 4 as well because of the second reduction phase
	printf("lws: %ld, nvolumes/4: %d, nwg: %d\n", lws, reduce_volumes/4, nwg);

	size_t memsize = reduce_volumes*sizeof(float);
	const size_t nwg_mem = nwg*sizeof(float);

	cl_mem d_funcValues1, d_funcValues2;

	d_funcValues1 = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_funcValues1");
	d_funcValues2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE,
		nwg_mem, NULL,
		&err);
	ocl_check(err, "create buffer d_funcValues2");

	float *h_null = (float*) malloc(memsize);
	initializeVal(h_null, reduce_volumes);

	err = clEnqueueWriteBuffer(que, d_funcValues1, CL_TRUE, 0, memsize, h_null, 0, NULL, NULL);
	ocl_check(err, "initialize volumes buffer funcValues1");

	err = clEnqueueWriteBuffer(que, d_funcValues2, CL_TRUE, 0, nwg_mem, h_null, 0, NULL, NULL);
	ocl_check(err, "initialize volumes buffer funcValues2");

	free((void*)h_null);

	cl_event intcalc_evt, reduce_evt[2], read_evt;
	intcalc_evt = intcalc(intcalc_k, que, d_funcValues1, leftLimitX, leftLimitY, leftLimitZ, offX, offY, offZ, ncuboidsPerDim);
	// reduce datasize to nwg elements
	reduce_evt[0] = vol_reduction(reduce4_k, que, d_funcValues2, d_funcValues1, reduce_volumes/4,
		lws, nwg, intcalc_evt);
	// wrap up reduction
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
	
	double calculatedValue = offX*offY*offZ*sum_result;
	printf("\n(%s, %s)∫(%s, %s)∫(%s, %s)∫[%s] dxdydz = %g\n\n", argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[1], calculatedValue);
	if (argv[9]) check_accuracy(calculatedValue, strtod(argv[9], NULL));

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
			reduce_volumes, runtime_pass_ms, pass_bw_gbs,
			reduce_volumes/1.0e6/runtime_pass_ms);
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
		nvolumes, runtime_reduction_ms, nvolumes/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//double absErr = strtod(argv[7], NULL) - calculatedValue;
	//if(absErr < 0) absErr *= -1;
	//printf("\nTotal:%d;%g;%g\n", nvolumes, total_time_ms, absErr);

	clReleaseMemObject(d_funcValues1);
	clReleaseMemObject(d_funcValues2);

	clReleaseKernel(intcalc_k);
	clReleaseKernel(reduce4_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}