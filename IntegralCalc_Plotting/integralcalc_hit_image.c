//Definite integral calculator (positive area)
//Monte Carlo Hit or Miss with local max GPU computation
//Plotting random points in a plot.pam image:
//the point is red if it's under the curve, blue otherwise
//GPU plotting with OpenCL image API

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
cl_event montecarlo(cl_kernel montecarlo_k, cl_command_queue que, cl_mem d_hits, 
	float leftLimit, float rightLimit, float funcMax, cl_int _gws, cl_mem d_plot, 
	cl_uint seed1, cl_uint seed2, cl_event prev_evt){

	const size_t gws[] = { _gws, _gws };

	cl_event montecarlo_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_hits), &d_hits);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(rightLimit), &rightLimit);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(funcMax), &funcMax);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(d_plot), &d_plot);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed1), &seed1);
	ocl_check(err, "set montecarlo arg %d ", i-1);
	err = clSetKernelArg(montecarlo_k, i++, sizeof(seed2), &seed2);
	ocl_check(err, "set montecarlo arg %d ", i-1);

	err = clEnqueueNDRangeKernel(que, montecarlo_k, 1, NULL, gws, NULL,
		1, &prev_evt, &montecarlo_evt);
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

void createBlankImageHost(uchar * imgData, size_t n){
	for(int i=0; i<n; i+=4){
		imgData[i]=255;
		imgData[i+1]=255;
		imgData[i+2]=255;
		imgData[i+3]=255;
	}
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

	cl_kernel imginit_k = clCreateKernel(prog, "imginit", &err);
	ocl_check(err, "create kernel imginit");

	cl_kernel montecarlo_k = clCreateKernel(prog, "montecarlo_hit_image", &err);
	ocl_check(err, "create kernel montecarlo");
	
	cl_uint seed1 = time(0) & 134217727;
	cl_uint seed2 = rdtsc() & 134217727;

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
	int maxIter = reductionSteps*2;
	int k = 0;	//Current iteration
	float a = leftLimit;
	float b = rightLimit;
	float dist = (intervalSize)/(nsamples-1);
	float pk = intervalSize/2.0;
	cl_float2 prev_max, h_max;
	prev_max.x = 0.0f;
	prev_max.y = 0.0f;
	h_max.x = 1.0f;
	h_max.y = 1.0f;

	cl_mem d_v1, d_v2;
	cl_mem d_hits;

	int *h_null = (int*) malloc(sizeof(int));
	*h_null = 0;

	d_v1 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*_gws*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 2*nwg_reduce*sizeof(float), NULL, &err);
	ocl_check(err, "create buffer d_v2");
	d_hits = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
		sizeof(cl_int), h_null, &err);
	ocl_check(err, "create buffer d_hits");

	free(h_null);

	cl_event max_evt[maxIter];	//An event for each funcSamples call
	cl_event reduce_evt[2*maxIter];	//2 events for each reduction: first we reduce to nwg elements, then we deal with the rest
	cl_event read_max_evt[maxIter];
	cl_event imginit_evt, montecarlo_evt, readHits_evt, readPlot_evt;

	while(k<maxIter && (h_max.y != prev_max.y || k<2)){	//At least two iterations are needed: in the first one h_max might be equal to our arbitrary starting value
		prev_max.x = h_max.x;
		prev_max.y = h_max.y;
		max_evt[k] = funcSamples(funcSamples_k, que, d_v1, a, dist, _gws);
		reduce_evt[2*k] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v1, d_v2, lws_reduce, _gws, max_evt[k]);
		if (nwg_reduce == 1) reduce_evt[2*k+1] = reduce_evt[2*k];
		else reduce_evt[2*k+1] = reduceMax_lmem(reduceMax_lmem_k, que,
				d_v2, d_v2, lws_reduce, nwg_reduce/2, reduce_evt[2*k]);

		err = clEnqueueReadBuffer(que, d_v2, CL_TRUE, 0, sizeof(h_max), &h_max,
		1, reduce_evt+2*k+1, read_max_evt+k);
		ocl_check(err, "read max");
		pk = pk/2.0f;	//halves the current interval size, then create the new interval
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
		fprintf(stderr, "Error: max is less or equal to 0. Can't integrate a negative function.\n");
		exit(9);
	}

	h_max.y += 0.01f;	//Making the grid a tiny bit higher to be safe

	//We now have everything we need to create the base image
	const char *imageName = "plot.pam";
	struct imgInfo plotInfo;
	plotInfo.channels = 4;
	plotInfo.depth = 8;
	plotInfo.maxval = 0xff;
	//Min size is 512x512, the smaller side must be 512; the other one must be proportional to their ratio
	if(h_max.y < intervalSize){
		plotInfo.height = 512;
		plotInfo.width = ceil(512*((intervalSize/h_max.y)-0.01));	//1.0f/1.0f might result into 1.01f
	}
	else{
		plotInfo.width = 512;
		plotInfo.height = ceil(512*((h_max.y/intervalSize)-0.01));
	}
	//plotInfo.width = plotInfo.height = 512;
	plotInfo.data_size = plotInfo.width*plotInfo.height*plotInfo.channels;
	plotInfo.data = malloc(plotInfo.data_size);
	printf("Processing image %dx%d with data size %ld bytes\n", plotInfo.width, plotInfo.height, plotInfo.data_size);
	
	const cl_image_format fmt = {
		.image_channel_order = CL_RGBA,
		.image_channel_data_type = CL_UNSIGNED_INT8,
	};
	const cl_image_desc desc = {
		.image_type = CL_MEM_OBJECT_IMAGE2D,
		.image_width = plotInfo.width,
		.image_height = plotInfo.height,
	};
	cl_mem d_plot = clCreateImage(ctx,
		CL_MEM_WRITE_ONLY,
		&fmt, &desc,
		plotInfo.data,
		&err);
	ocl_check(err, "create plot image");

	imginit_evt = imginit(imginit_k, que, d_plot, plotInfo.width, plotInfo.height);
	montecarlo_evt = montecarlo(montecarlo_k, que, d_hits, leftLimit, rightLimit, 
		h_max.y, npoints/2, d_plot, seed1, seed2, imginit_evt);

	cl_int * h_hits = clEnqueueMapBuffer(que, d_hits, 
		CL_TRUE, CL_MAP_READ, 0, sizeof(cl_int), 1, &montecarlo_evt, &readHits_evt, &err);
	ocl_check(err, "read hits from device");

	size_t origin[3] = {0,0,0};
	size_t region[3] = {plotInfo.height, plotInfo.width, 1};
	err = clEnqueueReadImage(que, d_plot, CL_TRUE,
			origin, region,
			0, 0, plotInfo.data,
			1, &readHits_evt, &readPlot_evt);
	ocl_check(err, "reading image to host");

	err = save_pam(imageName, &plotInfo);
	if (err != 0) {
		fprintf(stderr, "error writing %s\n", imageName);
		exit(1);
	}
	else printf("Successfully created plot image %s in the current directory\n", imageName);
	printf("The number of hits is %d\n\n", *h_hits);

	double calculatedValue = computeIntegral(h_max.y, intervalSize, *h_hits, npoints);
	printf("(%s, %s)∫%s dx = %f\n\n", argv[2], argv[3], argv[1], calculatedValue);
	if (argv[5]) check_accuracy(calculatedValue, strtod(argv[5], NULL));

	double runtime_max_ms = total_runtime_ms(max_evt[0], read_max_evt[k-1]);
	double runtime_imginit_ms = runtime_ms(imginit_evt);
	double runtime_montecarlo_ms = runtime_ms(montecarlo_evt);
	double runtime_readHits_ms = runtime_ms(readHits_evt);
	double runtime_readPlot_ms = runtime_ms(readPlot_evt);
	double total_time_ms = runtime_max_ms + runtime_imginit_ms + runtime_montecarlo_ms + 
		runtime_readHits_ms + runtime_readPlot_ms;

	double max_bw_gbs = k*nsamples/1.0e6/runtime_max_ms;
	double imginit_bw_gbs = plotInfo.width*plotInfo.height/1.0e6/runtime_max_ms;
	double montecarlo_bw_gbs = (*h_hits * sizeof(cl_int) + npoints*sizeof(cl_int))/1.0e6/runtime_montecarlo_ms;
	double readHits_bw_gbs = sizeof(cl_int)/1.0e6/runtime_readHits_ms;
	double readPlot_bw_gbs = 2*npoints*sizeof(cl_int)/1.0e6/runtime_readPlot_ms;

	printf("max: %d points (float, float) in %gms: %g GB/s\n", k*nsamples, runtime_max_ms, max_bw_gbs);
	printf("imginit : %d points (float, float) in %gms: %g GB/s\n",
		npoints, runtime_imginit_ms, imginit_bw_gbs);
	printf("montecarlo : %d points (float, float) in %gms: %g GB/s\n",
		npoints, runtime_montecarlo_ms, montecarlo_bw_gbs);
	printf("read hits : 1 int in %gms: %g GB/s\n",
		runtime_readHits_ms, readHits_bw_gbs);
	printf("read plot data : %dx%d pixels in %gms: %g GB/s\n",
		plotInfo.width, plotInfo.height, runtime_readPlot_ms, readPlot_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);
	//printf("Total:%g;%g\n", intervalSize, runtime_max_ms);

	err = clEnqueueUnmapMemObject(que, d_hits, h_hits, 0, NULL, NULL);
	ocl_check(err, "unmap hits");
	clReleaseMemObject(d_hits);
	clReleaseMemObject(d_plot);
	clReleaseMemObject(d_v1);
	clReleaseMemObject(d_v2);

	clReleaseKernel(funcSamples_k);
	clReleaseKernel(reduceMax_lmem_k);
	clReleaseKernel(imginit_k);
	clReleaseKernel(montecarlo_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}