//Definite integral calculator (positive area)
//Monte Carlo Sample-Mean method
//Plotting random points in a plotMean.pam image:
//blue points are the one on the function graph, the couples (x, f(x)) for every random number x.
//the red line represents the integral mean
//GPU plotting in an uchar4 array

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../ocl_boiler.h"
#include "../pamalign.h"

static inline uint64_t rdtsc(void)
{
	uint64_t val;
	uint32_t h, l;
    __asm__ __volatile__("rdtsc" : "=a" (l), "=d" (h));
        val = ((uint64_t)l) | (((uint64_t)h) << 32);
        return val;
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

cl_event reduce_MinOrMax_lmem(cl_kernel reduce_lmem_k, cl_command_queue que,
	cl_mem d_v1, cl_mem d_v2, int _lws, int _gws, cl_event prev_evt){
	//printf("Reduce gws: %d\n\n", _gws);
	if (_gws < _lws) _lws = _gws;
	const size_t gws[] = { _gws };
	const size_t lws[] = { _lws };
	cl_event reduce_lmem_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reduce_lmem_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set reduce_lmem arg ", i-1);
	err = clSetKernelArg(reduce_lmem_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set reduce_lmem arg ", i-1);
	err = clSetKernelArg(reduce_lmem_k, i++, 2*sizeof(float)*lws[0], NULL);
	ocl_check(err, "set reduce_lmem arg", i-1);
	err = clEnqueueNDRangeKernel(que, reduce_lmem_k, 1, NULL, gws, lws,
		0, NULL, &reduce_lmem_evt);
	ocl_check(err, "enqueue reduce_lmem");
	return reduce_lmem_evt;
}

cl_event reduce_MinAndMax_lmem(cl_kernel reduceMinMax_lmem_k, cl_command_queue que,
	cl_mem d_v1, cl_mem d_v2, int _lws, int _gws, cl_event prev_evt){
	//printf("Reduce gws: %d\n\n", _gws);
	if (_gws < _lws) _lws = _gws;
	const size_t gws[] = { _gws };
	const size_t lws[] = { _lws };
	cl_event reduceMinMax_lmem_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(reduceMinMax_lmem_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set reduceMinMax_lmem arg ", i-1);
	err = clSetKernelArg(reduceMinMax_lmem_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set reduceMinMax_lmem arg ", i-1);
	err = clSetKernelArg(reduceMinMax_lmem_k, i++, 4*sizeof(float)*lws[0], NULL);
	ocl_check(err, "set reduceMinMax_lmem arg", i-1);
	err = clEnqueueNDRangeKernel(que, reduceMinMax_lmem_k, 1, NULL, gws, lws,
		0, NULL, &reduceMinMax_lmem_evt);
	ocl_check(err, "enqueue reduceMinMax_lmem");
	return reduceMinMax_lmem_evt;
}

cl_event imginit(cl_kernel imginit_k, cl_command_queue que, cl_mem d_plot, int plotWidth, int plotHeight){

	const size_t gws[] = { plotWidth, plotHeight };

	cl_event imginit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(imginit_k, i++, sizeof(d_plot), &d_plot);
	ocl_check(err, "set imginit arg %d ", i-1);

	err = clEnqueueNDRangeKernel(que, imginit_k, 2, NULL, gws, NULL,
		0, NULL, &imginit_evt);
	ocl_check(err, "enqueue imginit");

	return imginit_evt;	
}

//Setting up the kernel to compute the hit number
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que, cl_mem d_sum, 
	float leftLimit, float rightLimit, float funcMin, float funcMax, cl_int _gws, 
	cl_mem d_plot, cl_uint seed1, cl_uint seed2, cl_int plotWidth, cl_int plotHeight){

	const size_t gws[] = { _gws };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_sum), &d_sum);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimit), &rightLimit);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(funcMin), &funcMin);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(funcMax), &funcMax);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_plot), &d_plot);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed1), &seed1);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed2), &seed2);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(plotWidth), &plotWidth);
	ocl_check(err, "set montecarlo arg %d", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(plotHeight), &plotHeight);
	ocl_check(err, "set montecarlo arg %d", i-1);

	err = clEnqueueNDRangeKernel(que, montecarlo_k, 1, NULL, gws, NULL,
		0, NULL, &montecarlo_evt);
	ocl_check(err, "enqueue montecarlo");

	return montecarlo_evt;	
}

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

	return reduce4_evt;
}

double computeIntegral(const double sumValue, const double intervalSize, const int npoints){
 	return intervalSize * (sumValue / npoints);
}

void check_accuracy(const double calculatedValue, const double expectedValue){
	printf("Expected value is %f, calculated value is %f\n", expectedValue, calculatedValue);
	double absErr = expectedValue - calculatedValue;
	if(absErr < 0) absErr *= -1;
	printf("The absolute error is %f\n", absErr);
}

void createBlankImage(uchar * imgData, size_t n){
	for(int i=0; i<n; i+=4){
		imgData[i]=255;
		imgData[i+1]=255;
		imgData[i+2]=255;
		imgData[i+3]=255;
	}
}

int drawHorizontalLine(cl_uchar4 * imgData, int imgWidth, int imgHeight, float funcMin, float funcMax, float yValue, cl_uchar4 color){
	int img_y = imgHeight-1 - ((yValue-funcMin)/(funcMax-funcMin))*(imgHeight-1);
	if (img_y <= 0 || img_y >= imgHeight){
		fprintf(stderr, "Can't draw y=%f: img_y=%d\n", yValue, img_y);
		return -1;
	}
	for(int i=0; i<imgWidth; i++){
		imgData[img_y*imgWidth+i]=color;
	}
	return 0;
}

int drawVerticalLine(cl_uchar4 * imgData, int imgWidth, int imgHeight, float leftLimit, float rightLimit, float xValue, cl_uchar4 color){
	int img_x = ((xValue-leftLimit)/(rightLimit-leftLimit))*(imgWidth-1);
	if (img_x <= 0 || img_x >= imgWidth){
		fprintf(stderr, "Can't draw x=%f: img_x=%d\n", xValue, img_x);
		return -1;
	}
	for(int i=0; i<imgWidth; i++){
		imgData[i*imgWidth+img_x]=color;
	}
	return 0;
}

int main(int argc, char* argv[]){

	int npoints = 128; 
	float leftLimit = 0.0f;
	float rightLimit = 1.0f;
	if(argc <= 4){
		printf("This program computes the area in an interval\n under the curve of a function.\n");
		fprintf(stderr, "Use: %s f(x) leftLimit rightLimit npoints [expectedValue]\n", argv[0]);
		printf("Examples: %s 'sqrt(1-x*x)' 0.0 1.0 1024 0.78539825\n", argv[0]);
		printf("%s 'log(1+x)/x' 0.0 1.0 4096 0.822465644\n", argv[0]);
		printf("%s 'exp(-x*x)' 0.5 6.3 4096 0.424946\n", argv[0]);
		exit(0);
	}
	else{
		npoints = atoi(argv[4]);
		if(npoints & 1){
			fprintf(stderr, "npoints must be a multiple of 2. Rounding up to the nearest multiple.\n");
			npoints = round_mul_up(npoints, 2);
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

	cl_kernel reduceMin_lmem_k = clCreateKernel(prog, "reduceMin_lmem", &err);
	ocl_check(err, "create kernel reduceMin_lmem");

	cl_kernel reduce_MinAndMax_lmem_k = clCreateKernel(prog, "reduceMinAndMax_lmem", &err);
	ocl_check(err, "create kernel reduceMinAndMax_lmem");

	cl_kernel reduce_MinAndMax_nwg_lmem_k = clCreateKernel(prog, "reduceMinAndMax_nwg_lmem", &err);
	ocl_check(err, "create kernel reduceMinAndMax_nwg_lmem");

	cl_kernel imginit_k = clCreateKernel(prog, "imginit_buf", &err);
	ocl_check(err, "create kernel imginit");

	cl_kernel montecarlo_k = clCreateKernel(prog, "montecarlo_mean_plot", &err);
	ocl_check(err, "create kernel montecarlo");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	cl_uint seed1 = time(0) & 134217727;
	cl_uint seed2 = rdtsc() & 134217727;

	size_t sum_lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(sum_lws), &sum_lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int sum_nwg;
	if (npoints/8 < sum_lws){
		sum_nwg = 1;
		sum_lws = npoints/8;
	}
	else sum_nwg = round_mul_up(npoints/8, sum_lws)/sum_lws;
	//if (nwg<2 && nwg != 1) nwg=2;
	printf("sum_lws: %ld, npoints/8: %d, sum_nwg: %d\n", sum_lws, npoints/8, sum_nwg);
	size_t memsize = (npoints/2)*sizeof(float);
	const size_t nwg_mem = sum_nwg*sizeof(float);

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
	
	size_t lws_max;
	err = clGetKernelWorkGroupInfo(reduceMax_lmem_k, d, CL_KERNEL_WORK_GROUP_SIZE, 
		sizeof(lws_max), &lws_max, NULL);
	ocl_check(err, "Max lws for reduction");
	size_t gws_max = 131072;

	double intervalSize = rightLimit-leftLimit;

	//Values for funcSamples/reduceMax_lmem
	int reductionSteps = 4+(int)ceil(log2(ceil(intervalSize)));
	int nsamples = 2<<(reductionSteps-1);
	int _gws = nsamples/2;
	if(_gws > gws_max){
		printf("Gws from %d to max possible value %ld\n", _gws, gws_max);
		_gws = gws_max;
	}
	int lws_reduce = 2<<(reductionSteps-4);
	if (lws_reduce > lws_max){
		printf("Lws from %d to max possible value %ld\n", lws_reduce, lws_max);
		lws_reduce = lws_max;
	}
	int nwg_reduce = _gws/lws_reduce;
	printf("lws=%d, nsamples = %d, _gws=%d\n\n", lws_reduce, nsamples, _gws);
	int maxIter = reductionSteps*2 - 1;	//Max steps for the solo max/min search, the first one will be combined
	int kmax = 0;	//Current iteration
	float a = leftLimit;
	float b = rightLimit;
	float dist = (intervalSize)/(nsamples-1);
	float pk = intervalSize/2.0;
	cl_float2 prev_max, h_max;
	prev_max.x = 0.0f;
	prev_max.y = 0.0f;
	h_max.x = 1.0f;
	h_max.y = 1.0f;

	int kmin = 0;
	cl_float4 first_minmax;
	cl_float2 prev_min, h_min;
	prev_min.x = 0.0f;
	prev_min.y = 0.0f;
	h_min.x = 1.0f;
	h_min.y = 1.0f;

	cl_mem d_v1, d_v2;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*_gws*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*nwg_reduce*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v2");

	cl_event max_evt[maxIter-1];	//An event for each funcSamples call
	cl_event min_evt[maxIter-1];
	cl_event reduceMax_evt[2*(maxIter-1)];	//2 events for each reduction: first we reduce to nwg elements, then we deal with the rest
	cl_event reduceMin_evt[2*(maxIter-1)];
	cl_event read_max_evt[maxIter-1];
	cl_event read_min_evt[maxIter-1];

	//events for the first combined search
	cl_event minmax_evt;
	cl_event reduceMinMax_evt[2];
	cl_event read_minmax_evt;

	cl_event montecarlo_evt, reduce_evt[2], initPlot_evt, readSum_evt, readPlot_evt;	//events for later

	//Combined min/max search
	minmax_evt = funcSamples(funcSamples_k, que, d_v1, a, dist, _gws);
		reduceMinMax_evt[0] = reduce_MinAndMax_lmem(reduce_MinAndMax_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, minmax_evt);
		if (nwg_reduce == 1) reduceMinMax_evt[1] = reduceMinMax_evt[0];
		else reduceMinMax_evt[1] = reduce_MinAndMax_lmem(reduce_MinAndMax_nwg_lmem_k, que,	//different kernel
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduceMinMax_evt[0]);

	err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(first_minmax), &first_minmax,
			1, reduceMinMax_evt+1, &read_minmax_evt);
			ocl_check(err, "read minmax");
	h_max.x = first_minmax.x;
	h_max.y = first_minmax.y;
	h_min.x = first_minmax.z;
	h_min.y = first_minmax.w;

	printf("Max and min values for the first iteration: max=%f, min=%f\n", h_max.y, h_min.y);

	//Create new interval, then get samples in the new interval to search the max
	while(kmax<maxIter && (h_max.y != prev_max.y || kmax<1)){	//At least two iterations are needed (the first one is the combined search)
		
		pk = pk/2.0f;	//halves the current interval size, then create the new interval
		if (h_max.x-pk > leftLimit) a = h_max.x-pk;
		else a = leftLimit;
		if (h_max.x+pk < rightLimit) b = h_max.x+pk;
		else b = rightLimit;
		dist = (b-a)/(nsamples-1);
		prev_max.x = h_max.x;
		prev_max.y = h_max.y;
		max_evt[kmax] = funcSamples(funcSamples_k, que, d_v1, a, dist, _gws);
		reduceMax_evt[2*kmax] = reduce_MinOrMax_lmem(reduceMax_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, max_evt[kmax]);
		if (nwg_reduce == 1) reduceMax_evt[2*kmax+1] = reduceMax_evt[2*kmax];
		else reduceMax_evt[2*kmax+1] = reduce_MinOrMax_lmem(reduceMax_lmem_k, que,
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduceMax_evt[2*kmax]);

		err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(h_max), &h_max,
		1, reduceMax_evt+2*kmax+1, read_max_evt+kmax);
		ocl_check(err, "read max");
		++kmax;

	}
	printf("Max computed in %d iterations vs maxIter=%d\n", kmax, maxIter);	//we have to consider the combined search
	printf("The max in [%f,%f] is f(%f) = %f\n", leftLimit, rightLimit, h_max.x, h_max.y);

	//restore interval size before calculating the min
	pk=intervalSize/2.0;

	//Create new interval, then get samples in the new interval to search the min
	while(kmin<maxIter && (h_min.y != prev_min.y || kmin<1)){	//At least two iterations are needed (the first one is the combined search)
		
		pk = pk/2.0f;	//halves the current interval size, then create the new interval
		if (h_min.x-pk > leftLimit) a = h_min.x-pk;
		else a = leftLimit;
		if (h_min.x+pk < rightLimit) b = h_min.x+pk;
		else b = rightLimit;
		dist = (b-a)/(nsamples-1);
		prev_min.x = h_min.x;
		prev_min.y = h_min.y;
		min_evt[kmin] = funcSamples(funcSamples_k, que, d_v1, a, dist, _gws);
		reduceMin_evt[2*kmin] = reduce_MinOrMax_lmem(reduceMin_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, min_evt[kmin]);
		if (nwg_reduce == 1) reduceMin_evt[2*kmin+1] = reduceMin_evt[2*kmin];
		else reduceMin_evt[2*kmin+1] = reduce_MinOrMax_lmem(reduceMin_lmem_k, que,
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduceMin_evt[2*kmin]);

		err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(h_min), &h_min,
		1, reduceMin_evt+2*kmin+1, read_min_evt+kmin);
		ocl_check(err, "read max");
		++kmin;

	}
	printf("Min computed in %d iterations vs maxIter=%d\n", kmin, maxIter);
	printf("The min in [%f,%f] is f(%f) = %f\n\n", leftLimit, rightLimit, h_min.x, h_min.y);

	//We now have everything we need to create the base image
	const double funcHeight = h_max.y - h_min.y;
	const char *imageName = "plotMean.pam";
	struct imgInfo plotInfo;
	plotInfo.channels = 4;
	plotInfo.depth = 8;
	plotInfo.maxval = 0xff;
	//Min size is 512x512, the smaller side must be 512; the other one must be proportional to their ratio
	if(funcHeight < intervalSize){
		plotInfo.height = 512;
		plotInfo.width = ceil(512*((intervalSize/funcHeight)-0.01));	//1.0f/1.0f might result into 1.01f
	}
	else{
		plotInfo.width = 512;
		plotInfo.height = ceil(512*((funcHeight/intervalSize)-0.01));
	}
	//plotInfo.width = plotInfo.height = 512;
	plotInfo.data_size = plotInfo.width*plotInfo.height*plotInfo.channels;
	plotInfo.data = malloc(plotInfo.data_size);;
	printf("Processing image %dx%d with data size %ld bytes\n", plotInfo.width, plotInfo.height, plotInfo.data_size);

	cl_mem d_plot = clCreateBuffer(ctx,
		CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
		plotInfo.data_size, NULL,
		&err);
	ocl_check(err, "create buffer d_plot");

	initPlot_evt = imginit(imginit_k, que, d_plot, plotInfo.width, plotInfo.height);

	montecarlo_evt = montecarlo(montecarlo_k, que, d_sum1, leftLimit, rightLimit, 
		h_min.y, h_max.y, npoints/2, d_plot, seed1, seed2, plotInfo.width, plotInfo.height);

	reduce_evt[0] = sum_reduction(reduce4_k, que, d_sum2, d_sum1, npoints/8,
		sum_lws, sum_nwg, montecarlo_evt);
	// concludo
	if (sum_nwg > 1) {
		reduce_evt[1] = sum_reduction(reduce4_k, que, d_sum2, d_sum2, round_mul_up(sum_nwg,4)/4,
			sum_lws, 1, reduce_evt[0]);
	} else {
		reduce_evt[1] = reduce_evt[0];
	}

	float sum_result;
	err = clEnqueueReadBuffer(que, d_sum2, CL_TRUE, 0, sizeof(sum_result), &sum_result,
		1, reduce_evt + 1, &readSum_evt);
	ocl_check(err, "read result");

	printf("The sum is %f\n\n", sum_result);

	plotInfo.data = clEnqueueMapBuffer(que, d_plot, CL_TRUE,
		CL_MAP_READ,
		0, plotInfo.data_size,
		1, &readSum_evt, &readPlot_evt, &err);
	ocl_check(err, "enqueue map d_plot");

	cl_uchar4 black = { .x = 0, .y = 0, .z = 0, .w = 255 };
	cl_uchar4 red = { .x = 255, .y = 0, .z = 0, .w = 255 };
	
	//integral mean line
	drawHorizontalLine((cl_uchar4*)plotInfo.data, plotInfo.width, plotInfo.height,	//draw mean line 
		h_min.y, h_max.y, sum_result/npoints, red);

	//x axis
	drawHorizontalLine((cl_uchar4*)plotInfo.data, plotInfo.width, plotInfo.height,	//draw x axis
		h_min.y, h_max.y, 0, black);

	//y axis
	drawVerticalLine((cl_uchar4*)plotInfo.data, plotInfo.width, plotInfo.height,	//draw y axis
		leftLimit, rightLimit, 0, black);

	err = save_pam(imageName, &plotInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("Successfully created plot image %s in the current directory\n", imageName);
	printf("The sum is %f\n\n", sum_result);

	double calculatedValue = computeIntegral(sum_result, intervalSize, npoints);
	printf("(%s, %s)∫%s dx = %g\n\n", argv[2], argv[3], argv[1], calculatedValue);
	if (argv[5]) check_accuracy(calculatedValue, strtod(argv[5], NULL));

	double runtime_minmax_ms = total_runtime_ms(minmax_evt, read_minmax_evt);
	double runtime_max_ms = total_runtime_ms(max_evt[0], read_max_evt[kmax-1]);
	double runtime_min_ms = total_runtime_ms(min_evt[0], read_min_evt[kmin-1]);
	double runtime_initPlot_ms = runtime_ms(initPlot_evt);
	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	const double runtime_reduction_ms = total_runtime_ms(reduce_evt[0], reduce_evt[1]);
	double runtime_readSum_ms = runtime_ms(readSum_evt);
	double runtime_readPlot_ms = runtime_ms(readPlot_evt);
	double total_time_ms = total_runtime_ms(max_evt[0], readSum_evt);

	double minmax_bw_gbs = nsamples/1.0e6/runtime_minmax_ms;
	double max_bw_gbs = kmax*nsamples/1.0e6/runtime_max_ms;
	double min_bw_gbs = kmin*nsamples/1.0e6/runtime_min_ms;
	double initPlot_bw_gbs = plotInfo.data_size/1.0e6/runtime_initPlot_ms;
	double montecarlo_bw_gbs = (memsize + 4*npoints*sizeof(uchar))/1.0e6/runtime_montecarlo_ms;
	double readSum_bw_gbs = sizeof(float)/1.0e6/runtime_readSum_ms;
	double readPlot_bw_gbs = plotInfo.data_size/1.0e6/runtime_readPlot_ms;

	printf("minmax (combined first step): %d points (float, float) in %gms: %g GB/s\n", nsamples, runtime_minmax_ms, minmax_bw_gbs);
	printf("max: %d points (float, float) in %gms: %g GB/s\n", kmax*nsamples, runtime_max_ms, max_bw_gbs);
	printf("min: %d points (float, float) in %gms: %g GB/s\n", kmin*nsamples, runtime_min_ms, min_bw_gbs);
	printf("init plot: %ld uchar in %gms: %g GB/s\n", plotInfo.data_size, runtime_initPlot_ms, initPlot_bw_gbs);
	printf("montecarlo : %d points (float, float) in %gms: %g GB/s\n",
		npoints, runtime_montecarlo_ms, montecarlo_bw_gbs);
	printf("reduce : %d float in %gms: %g GE/s\n",
		npoints, runtime_reduction_ms, npoints/1.0e6/runtime_reduction_ms);
	printf("read sum : 1 float in %gms: %g GB/s\n",
		runtime_readSum_ms, readSum_bw_gbs);
	printf("read plot data : %ld uchar in %gms: %g GB/s\n",
		plotInfo.data_size, runtime_readPlot_ms, readPlot_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//printf("Total:%g;%g\n", intervalSize, runtime_max_ms);

	err = clEnqueueUnmapMemObject(que, d_plot, plotInfo.data, 0, NULL, NULL);
	ocl_check(err, "unmap plot");
	clReleaseMemObject(d_sum1);
	clReleaseMemObject(d_sum2);
	clReleaseMemObject(d_plot);
	clReleaseMemObject(d_v1);
	clReleaseMemObject(d_v2);

	clReleaseKernel(funcSamples_k);
	clReleaseKernel(reduce_MinAndMax_lmem_k);
	clReleaseKernel(reduce_MinAndMax_nwg_lmem_k);
	clReleaseKernel(reduceMax_lmem_k);
	clReleaseKernel(reduceMin_lmem_k);
	clReleaseKernel(reduce4_k);
	clReleaseKernel(montecarlo_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}