//Definite integral calculator (positive area)
//Monte Carlo Hit or Miss with local max GPU computation

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
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
			if(linectr == 4 || linectr == 9 || linectr == 14){
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

//Setting up the kernel to retrieve function values from uniform samples
cl_event funcSamples(cl_kernel funcSamples_k, cl_command_queue que, cl_mem d_v1, 
	float leftLimit, float dist, int _gws){

	const size_t gws[] = { _gws };
	cl_event funcSamples_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(funcSamples_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clSetKernelArg(funcSamples_k, i++, sizeof(dist), &dist);
	ocl_check(err, "set funcSamples arg ", i-1);
	err = clEnqueueNDRangeKernel(que, funcSamples_k, 1, NULL, gws, NULL,
		0, NULL, &funcSamples_evt);
	ocl_check(err, "enqueue funcSamples");
	return funcSamples_evt;
}

cl_event reduceMax_lmem(cl_kernel reduceMax_lmem_k, cl_command_queue que,
	cl_mem d_v1, cl_mem d_v2, int _lws, int _gws, cl_event prev_evt){
	//printf("Reduce gws: %d\n\n", _gws);
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
	err = clSetKernelArg(reduceMax_lmem_k, i++, 2*sizeof(float)*lws[0], NULL);
	ocl_check(err, "set reduce4 arg", i-1);
	err = clEnqueueNDRangeKernel(que, reduceMax_lmem_k, 1, NULL, gws, lws,
		0, NULL, &reduceMax_lmem_evt);
	ocl_check(err, "enqueue reduceMax_lmem");
	return reduceMax_lmem_evt;
}

//Setting up the kernel to compute the hit number
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que, cl_mem d_hits, 
	float leftLimit, float rightLimit, float funcMax, 
	cl_uint seed1, cl_uint seed2, cl_uint seed3, cl_uint seed4, cl_int _gws){

	const size_t gws[] = { _gws };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_hits), &d_hits);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set montecarlo arg ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimit), &rightLimit);
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

//The area under the curve is approximately the area of the grid times the probability
double computeIntegral(const float c, const double intervalSize, const int hits, const int npoints){
 	return c*intervalSize*((double)hits / (double)npoints);
}

void check_accuracy(const double calculatedValue, const double expectedValue){
	printf("Expected value is %f, calculated value is %f\n", expectedValue, calculatedValue);
	double absErr = expectedValue - calculatedValue;
	if(absErr < 0) absErr *= -1;
	printf("The absolute error is %f\n", absErr);
}

int main(int argc, char* argv[]){

	int npoints = 128; 
	float leftLimit = 0.0f;
	float rightLimit = 1.0f;
	if(argc <= 4){
		printf("This program computes the area in an interval\n under the curve of a function in the first quadrant.\n");
		fprintf(stderr, "Use: %s f(x) leftLimit rightLimit npoints [expectedValue]\n", argv[0]);
		printf("Examples: %s 'sqrt(1-x*x)' 0.0 1.0 1024 0.78539825\n", argv[0]);
		printf("%s 'exp(-x*x)' 0.5 6.3 4096 0.424946\n", argv[0]);
		exit(0);
	}
	else{
		npoints = atoi(argv[4]);
		if(npoints & 7){
			fprintf(stderr, "npoints must be a multiple of 8. Rounding up to the nearest multiple.\n");
			npoints = round_mul_up(npoints, 8);
		}
		replaceFunction(argv[1]);
		leftLimit = strtof(argv[2], NULL);
		rightLimit = strtof(argv[3], NULL);
		if(leftLimit > rightLimit){
			fprintf(stderr, "The left limit is bigger than the right one. Swapping the limits.\n");
			float temp = leftLimit;
			leftLimit = rightLimit;
			rightLimit = temp;
		}
	}

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("montecarlo.ocl", ctx, d);
	cl_int err;

	cl_kernel funcSamples_k = clCreateKernel(prog, "funcSamples4", &err);
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
	
	//A way to get the max supported gws of the device, kind of irrelevant since most GPUs nowadays use 64 bit addresses

	//int addressBits;
	//err = clGetDeviceInfo(d, CL_DEVICE_ADDRESS_BITS, sizeof(addressBits), &addressBits, NULL);
	//printf("Address bits: %d, max = %lld\n", addressBits, (2<<(addressBits-1))-1);

	size_t gws_max = 131072;	//fixed max gws, no point in getting even more samples in most cases

	double intervalSize = rightLimit-leftLimit;

	//Values for funcSamples/reduceMax_lmem
	int reductionSteps = 4+(int)ceil(log2(ceil(intervalSize)));
	int nsamples = 2<<(reductionSteps-1);
	int _gws = nsamples/2;
	if(_gws > gws_max){
		printf("ReduceMax gws from %d to max possible value %ld\n", _gws, gws_max);
		_gws = gws_max;
	}
	int lws_reduce = 2<<(reductionSteps-4);
	if (lws_reduce > lws_max){
		printf("ReduceMax lws from %d to max possible value %ld\n", lws_reduce, lws_max);
		lws_reduce = lws_max;
	}
	int nwg_reduce = _gws/lws_reduce;
	printf("lws=%d, nsamples = %d, _gws=%d\n\n", lws_reduce, nsamples, _gws);
	int maxIter = reductionSteps*2;
	int k = 0;	//Current iteration
	float a = leftLimit;
	float b = rightLimit;
	float dist = (intervalSize)/(nsamples-1);
	float pk = intervalSize/2.0;	//half the size of the interval from which we get the samples
	cl_float2 prev_max, h_max;	//host float2 structs needed to retrieve the couples (x, f(x)) and save the max computed in the previous iteration
	prev_max.x = 0.0f;
	prev_max.y = 0.0f;
	h_max.x = 1.0f;
	h_max.y = 1.0f;

	cl_mem d_v1, d_v2;	//buffers needed for saving the samples and for the reduction
	cl_mem d_hits;

	int *h_null = (int*) malloc(sizeof(int));	//needed to initialize hits on device
	*h_null = 0;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*_gws*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*nwg_reduce*sizeof(float), NULL, &err);
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

	while(k<maxIter && (h_max.y != prev_max.y || k<2)){	//At least two iterations are needed: in the first one h_max might be equal to our arbitrary starting value
		prev_max.x = h_max.x;
		prev_max.y = h_max.y;
		max_evt[k] = funcSamples(funcSamples_k, que, d_v1, a, dist, _gws);	//retrieve the couples (x, f(x)) from the current interval
		reduce_evt[2*k] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, max_evt[k]);
		if (nwg_reduce == 1) reduce_evt[2*k+1] = reduce_evt[2*k];
		else reduce_evt[2*k+1] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduce_evt[2*k]);

		err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(h_max), &h_max,
		1, reduce_evt+2*k+1, read_max_evt+k);
		ocl_check(err, "read max");
		pk = pk/2.0f;	//halves the current interval, then create the new one
		if (h_max.x-pk > leftLimit) a = h_max.x-pk;
		else a = leftLimit;
		if (h_max.x+pk < rightLimit) b = h_max.x+pk;
		else b = rightLimit;
		dist = (b-a)/(nsamples-1);
		++k;

	}
	printf("Max computed in %d iterations vs maxIter=%d\n", k, maxIter);
	printf("The max in [%f,%f] is f(%f) = %f\n", leftLimit, rightLimit, h_max.x, h_max.y);

	if(h_max.y <= 0.0f){
		fprintf(stderr, "Error: max is less or equal to 0.\nCan't integrate a negative function with this method.\n");
		exit(9);
	}

	h_max.y += 0.01f;	//Making the grid a tiny bit higher to be safe

	montecarlo_evt = montecarlo(montecarlo_k, que, d_hits, leftLimit, rightLimit, h_max.y, seed1, seed2, seed3, seed4, npoints/8);

	cl_int * h_hits = clEnqueueMapBuffer(que, d_hits, 
		CL_TRUE, CL_MAP_READ, 0, sizeof(cl_int), 1, &montecarlo_evt, &read_evt, &err);
	ocl_check(err, "read hits from device");

	printf("The number of hits is %d\n\n", *h_hits);

	double calculatedValue = computeIntegral(h_max.y, intervalSize, *h_hits, npoints);
	printf("(%s, %s)∫%s dx = %f\n\n", argv[2], argv[3], argv[1], calculatedValue);
	if (argv[5]) check_accuracy(calculatedValue, strtod(argv[5], NULL));
	
	double runtime_max_ms = total_runtime_ms(max_evt[0], read_max_evt[k-1]);
	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	double runtime_read_ms = runtime_ms(read_evt);
	double total_time_ms = runtime_max_ms + runtime_montecarlo_ms + runtime_read_ms;

	double max_bw_gbs = k*nsamples*sizeof(float)/1.0e6/runtime_max_ms;
	double montecarlo_bw_gbs = (npoints/8)*sizeof(cl_int)/1.0e6/runtime_montecarlo_ms;
	double read_bw_gbs = sizeof(cl_int)/1.0e6/runtime_read_ms;

	printf("max: %d points (float, float) in %gms: %g GB/s\n", k*nsamples, runtime_max_ms, max_bw_gbs);
	printf("montecarlo : %d points (float, float) in %gms: %g GB/s\n",
		npoints, runtime_montecarlo_ms, montecarlo_bw_gbs);
	printf("read : hits (int) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	
	//Leftovers to print the formatted information needed to do plots
	//printf("Total:%g;%g\n", intervalSize, runtime_max_ms);

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