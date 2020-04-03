//Double integral calculator
//Monte Carlo Method "Hit or Miss" computing function local max
//NOTE: might not work with improper integrals and won't consider the volume under the xy plane

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

//Setting up the kernel to uniformly get samples from the function
cl_event funcSamples3D(cl_kernel funcSamples_k, cl_command_queue que, cl_mem d_v1, 
	float leftLimitX, float leftLimitY, float offX, float offY, int nsamplesPerDim){

	const size_t gws[] = { nsamplesPerDim, nsamplesPerDim };
	cl_event funcSamples_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(funcSamples_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(leftLimitX), &leftLimitX);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(leftLimitY), &leftLimitY);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(offX), &offX);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(offY), &offY);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clEnqueueNDRangeKernel(que, funcSamples_k, 2, NULL, gws, NULL,
		0, NULL, &funcSamples_evt);
	ocl_check(err, "enqueue funcSamples");
	return funcSamples_evt;

}

//Setting up the kernel to reduce the samples and find the highest point
cl_event reduceMax_lmem(cl_kernel reduceMax_lmem_k, cl_command_queue que,
	cl_mem d_v1, cl_mem d_v2, int _lws, int _gws, cl_event prev_evt){
	//printf("Reduce gws: %d, lws:%d\n\n", _gws, _lws);
	if (_gws < _lws) _lws = _gws;
	const size_t gws[] = { _gws };
	const size_t lws[] = { _lws };
	cl_event reduceMax_lmem_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reduceMax_lmem_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set reduceMax_lmem arg ", i-1);
	err = clSetKernelArg(reduceMax_lmem_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set reduceMax_lmem arg ", i-1);
	err = clSetKernelArg(reduceMax_lmem_k, i++, 4*sizeof(float)*lws[0], NULL);
	ocl_check(err, "set reduce4 arg", i-1);
	err = clEnqueueNDRangeKernel(que, reduceMax_lmem_k, 1, NULL, gws, lws,
		0, NULL, &reduceMax_lmem_evt);
	ocl_check(err, "enqueue reduceMax_lmem");
	return reduceMax_lmem_evt;
}

//Setting up the kernel that generates random points in 3D space and checks if they are under the curve
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que, cl_mem d_hits, 
	float leftLimitX, float rightLimitX, float leftLimitY, float rightLimitY, 
	float funcMax, cl_uint seed1, cl_uint seed2, cl_uint seed3, cl_uint seed4, int npoints){

	const size_t gws[] = { npoints/8 };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_hits), &d_hits);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimitX), &leftLimitX);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimitX), &rightLimitX);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimitY), &leftLimitY);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimitY), &rightLimitY);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(funcMax), &funcMax);
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

//The volume under the curve is approximately the volume of the grid times the probability
double computeIntegral(const float c, const double rectArea, const int hits, const int npoints){
 	return c*rectArea*((double)hits / (double)npoints);
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
	if(argc <= 6){
		printf("This program computes the volume enclosed by a rectangle\nunder the curve of a two-variable function\n");
		fprintf(stderr, "Use: %s f(x, y) leftLimitX rightLimitX leftLimitY rightLimitY npoints [expectedValue]\n", argv[0]);
		printf("Examples: %s 'x*x + y*y' 0.0 1.0 0.0 1.0 1024 0.666666666\n", argv[0]);
		printf("%s 'x*x + 4*y' 11 14 7 10 65536 1719\n", argv[0]);
		printf("%s 'x*sin(y)' 0 2 0 1.570796 131072 2\n", argv[0]);		
		exit(0);
	}
	else{
		npoints = atoi(argv[6]);
			if(npoints & 7){
				npoints = round_mul_up(npoints, 8);
				fprintf(stderr, "npoints must be a multiple of 8. Rounding up to %d\n", npoints);
		}
		leftLimitX = strtod(argv[2], NULL);
		rightLimitX = strtod(argv[3], NULL);
		if(leftLimitX > rightLimitX){
			fprintf(stderr, "The left limit is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimitX;
			leftLimitX = rightLimitX;
			rightLimitX = temp;
		}
		leftLimitY = strtod(argv[4], NULL);
		rightLimitY = strtod(argv[5], NULL);
		if(leftLimitY > rightLimitY){
			fprintf(stderr, "The left limit is bigger than the right one. Swapping the limits.\n");
			double temp = leftLimitY;
			leftLimitY = rightLimitY;
			rightLimitY = temp;
		}

	}
	double rectArea = (rightLimitX-leftLimitX)*(rightLimitY-leftLimitY);
	replaceFunction(argv[1]);

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;

	cl_kernel funcSamples_k = clCreateKernel(prog, "funcSamples3D", &err);
	ocl_check(err, "create kernel funcSamples");

	cl_kernel reduceMax_lmem_k = clCreateKernel(prog, "reduceMax_lmem", &err);
	ocl_check(err, "create kernel reduceMax_lmem");

	cl_kernel montecarlo_k = clCreateKernel(prog, "montecarlo_hit", &err);
	ocl_check(err, "create kernel montecarlo");
	
	//seeds for the edited MWC64X
	cl_uint seed1, seed2, seed3, seed4;
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	printf("Seeds: %d, %d, %d, %d\n", seed1, seed2, seed3, seed4);

	size_t lws_max;
	err = clGetKernelWorkGroupInfo(reduceMax_lmem_k, d, CL_KERNEL_WORK_GROUP_SIZE, 
		sizeof(lws_max), &lws_max, NULL);
	ocl_check(err, "Max lws for reduction");
	//size_t gws_max = 131072;

	size_t memsize = 4*npoints*sizeof(float);

	//Values for funcSamples/reduceMax_lmem
	int reductionSteps = 2+(int)ceil(log2(ceil(rectArea)));
	int nsamplesPerDim = 2<<(reductionSteps-1);
	int _gws = (nsamplesPerDim*nsamplesPerDim)/2;
	
	int lws_reduce = 2<<(reductionSteps-2);
	if (lws_reduce > lws_max){
		printf("Lws from %d to max possible value %ld\n", lws_reduce, lws_max);
		lws_reduce = lws_max;
	}
	int nwg_reduce = _gws/lws_reduce;
	printf("lws=%d, nsamplesPerDim = %d, _gws=%d\n\n", lws_reduce, nsamplesPerDim, _gws);
	int maxIter = reductionSteps*2;
	int k = 0;	//Current iteration
	float current_leftLimitX = leftLimitX;
	float current_rightLimitX = rightLimitX;
	float current_leftLimitY = leftLimitY;
	float current_rightLimitY = rightLimitY;
	float offX = (rightLimitX-leftLimitX)/(nsamplesPerDim-1);
	float offY = (rightLimitY-leftLimitY)/(nsamplesPerDim-1);
	float pkX = (rightLimitX-leftLimitX)/2.0f;
	float pkY = (rightLimitY-leftLimitY)/2.0f;
	cl_float4 prev_max, h_max;
	prev_max.x = 0.0f;
	prev_max.y = 0.0f;
	prev_max.z = 0.0f;
	prev_max.w = 0.0f;
	h_max.x = 1.0f;
	h_max.y = 1.0f;
	h_max.z = 0.0f;
	h_max.w = 0.0f;

	cl_mem d_v1, d_v2;
	cl_mem d_hits;

	cl_int *h_null = (int*) malloc(sizeof(cl_int));
	*h_null = 0;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 8*_gws*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 8*nwg_reduce*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v2");
	d_hits = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR | CL_MEM_COPY_HOST_PTR,
		sizeof(cl_int), h_null, &err);
	ocl_check(err, "create buffer d_hits");

	free(h_null);

	cl_event max_evt[maxIter];	//An event for each funcSamples call
	cl_event reduce_evt[2*maxIter];	//2 events for each reduction: first we reduce to nwg elements, then we deal with the rest
	cl_event read_max_evt[maxIter];
	cl_event montecarlo_evt, read_evt;
	while(k<maxIter && (h_max.w != prev_max.w || k<2)){	//At least two iterations are needed: in the first one h_max might be equal to our arbitrary starting value
		prev_max.x = h_max.x;
		prev_max.y = h_max.y;
		prev_max.w = h_max.w;
		max_evt[k] = funcSamples3D(funcSamples_k, que, d_v1, current_leftLimitX, 
			current_leftLimitY, offX, offY, nsamplesPerDim);
		clWaitForEvents(1, max_evt+k);
		reduce_evt[2*k] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, max_evt[k]);
		clWaitForEvents(1, reduce_evt+2*k);
		if (nwg_reduce == 1) reduce_evt[2*k+1] = reduce_evt[2*k];
		else reduce_evt[2*k+1] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduce_evt[2*k]);
		err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(h_max), &h_max,
		1, reduce_evt+2*k+1, read_max_evt+k);
		ocl_check(err, "read max");
		clWaitForEvents(1, read_max_evt+k);
		pkX = pkX/2.0f;	//halves the current interval size, then create the new interval
		pkY = pkY/2.0f;

		if (h_max.x-pkX > leftLimitX) current_leftLimitX = h_max.x-pkX;
		else current_leftLimitX = leftLimitX;
		if (h_max.x+pkX < rightLimitX) current_rightLimitX = h_max.x+pkX;
		else current_rightLimitX = rightLimitX;
		if (h_max.y-pkY > leftLimitY) current_leftLimitY = h_max.y-pkY;
		else current_leftLimitY = leftLimitY;
		if (h_max.y+pkY < rightLimitY) current_rightLimitY = h_max.y+pkY;
		else current_rightLimitY = rightLimitY;

		offX = (current_rightLimitX-current_leftLimitX)/(nsamplesPerDim-1);
		offY = (current_rightLimitY-current_leftLimitY)/(nsamplesPerDim-1);

		++k;

	}
	printf("Max computed in %d iterations vs maxIter=%d\n", k, maxIter);
	printf("The max in the [%f, %f] x [%f, %f] is f(%f, %f) = %f\n", leftLimitX, rightLimitX, 
		leftLimitY, rightLimitY, h_max.x, h_max.y, h_max.w);
	h_max.w += 0.01f;	//Making the grid a tiny bit higher to be safe

	montecarlo_evt = montecarlo(montecarlo_k, que, d_hits, leftLimitX, rightLimitX, 
		leftLimitY, rightLimitY, h_max.w, seed1, seed2, seed3, seed4, npoints);

	cl_int * h_hits = clEnqueueMapBuffer(que, d_hits, 
		CL_TRUE, CL_MAP_READ, 0, sizeof(cl_int), 1, &montecarlo_evt, &read_evt, &err);
	ocl_check(err, "read hits from device");

	printf("The number of hits is %d\n\n", *h_hits);

	double calculatedValue = computeIntegral(h_max.w, rectArea, *h_hits, npoints);
	printf("(%s, %s)∫(%s, %s)∫[%s] dxdy = %f\n\n", argv[2], argv[3], argv[4], argv[5], argv[1], calculatedValue);
	if (argv[7]) check_accuracy(calculatedValue, strtod(argv[7], NULL));

	double runtime_max_ms = total_runtime_ms(max_evt[0], read_max_evt[k-1]);
	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	double runtime_read_ms = runtime_ms(read_evt);
	double total_time_ms = runtime_max_ms + runtime_montecarlo_ms + runtime_read_ms;

	double max_bw_gbs = k*nsamplesPerDim*nsamplesPerDim*sizeof(float)/1.0e6/runtime_max_ms;
	double montecarlo_bw_gbs = (npoints/8)*sizeof(cl_int)/1.0e6/runtime_montecarlo_ms;
	double read_bw_gbs = sizeof(cl_int)/1.0e6/runtime_read_ms;

	printf("max: %d points (float, float, float) in %gms: %g GB/s\n", k*_gws, runtime_max_ms, max_bw_gbs);
	printf("montecarlo : %d points (float, float, float) in %gms: %g GB/s\n",
		npoints, runtime_montecarlo_ms, montecarlo_bw_gbs);
	printf("read : hits (int) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//double absErr = strtod(argv[7], NULL) - calculatedValue;
	//if(absErr < 0) absErr *= -1;
	//printf("\nTotal:%d;%g;%g\n", npoints/8, total_time_ms, absErr);

	err = clEnqueueUnmapMemObject(que, d_hits, h_hits, 0, NULL, NULL);
	ocl_check(err, "unmap hits");
	clReleaseMemObject(d_hits);
	clReleaseMemObject(d_v1);
	clReleaseMemObject(d_v2);

	clReleaseKernel(funcSamples_k);
	clReleaseKernel(reduceMax_lmem_k);
	clReleaseKernel(montecarlo_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}