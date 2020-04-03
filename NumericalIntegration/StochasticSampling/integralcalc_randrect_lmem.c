//Definite integral calculator with the rectangle rule (stochastic sampling)
//With local memory

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <unistd.h>

#define CL_TARGET_OPENCL_VERSION 120
#define MAX 256
#include "../../ocl_boiler.h"

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

	code = fopen("intcalc.ocl", "r");
	temp = fopen("temp.txt", "w");

	while(!feof(code)){
		strcpy(str, "\0");
		fgets(str, MAX, code);
		if(!feof(code)){
			linectr++;
			if(linectr == 4 || linectr==9 || linectr==14){
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

//Setting up the kernel to fill a buffer with random numbers
cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que,
	cl_mem d_v, float leftLimit, float rightLimit, 
	cl_uint seed1, cl_uint seed2, cl_uint seed3, cl_uint seed4, int nsamples){

	const size_t gws[] = { nsamples/8 };

	cl_event vecinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(vecinit_k, i++, sizeof(d_v), &d_v);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(leftLimit), &leftLimit);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(rightLimit), &rightLimit);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(seed1), &seed1);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(seed2), &seed2);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(seed3), &seed3);
	ocl_check(err, "set vecinit arg ", i-1);
	err = clSetKernelArg(vecinit_k, i++, sizeof(seed4), &seed4);
	ocl_check(err, "set vecinit arg ", i-1);
	
	err = clEnqueueNDRangeKernel(que, vecinit_k, 1, NULL, gws, NULL,
		0, NULL, &vecinit_evt);
	ocl_check(err, "enqueue vecinit");

	return vecinit_evt;	
}

cl_event sortparallel(cl_kernel sortinit_k,cl_int _lws, cl_command_queue que,
	cl_mem d_v1, cl_int nels, cl_event init_event)
{
	const size_t workitem=nels; 
	const size_t gws[] = { round_mul_up(workitem,_lws ) };
	const size_t lws[] = {_lws };
	cl_event sortinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(sortinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set sortinit arg1", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set sortinit arg2", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set sortinit arg3", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(float)*_lws,NULL);
	ocl_check(err, "set sortinit localmemory arg",i-1);

	err = clEnqueueNDRangeKernel(que, sortinit_k, 1,
		NULL, gws, lws,
		1, &init_event, &sortinit_evt);
	
	ocl_check(err, "enqueue sortinit");

	return sortinit_evt;
}

cl_event sortparallelmerge(cl_kernel sortinit_k,cl_int _lws, cl_command_queue que,
	cl_mem d_v1,cl_mem d_vout, cl_int nels, cl_event init_event,cl_int current_merge_size)
{
	const size_t workitem=nels; 
	const size_t gws[] = { round_mul_up(workitem, _lws ) };
	const size_t lws[] = {_lws };
	printf("merge gws e workitem : %d | %u = %zu  %li\n", nels, _lws, gws[0],workitem);
	cl_event sortinit_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(sortinit_k, i++, sizeof(d_vout), &d_vout);
	ocl_check(err, "set mergeinit arg1", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set mergeinit arg3", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(nels), &nels);
	ocl_check(err, "set mergeeinit arg2", i-1);
	err = clSetKernelArg(sortinit_k, i++, sizeof(current_merge_size),&current_merge_size);
	ocl_check(err, "set mergeinit arg4",i-1);

	err = clEnqueueNDRangeKernel(que, sortinit_k, 1,
		NULL, gws, lws,
		1, &init_event, &sortinit_evt);
	
	ocl_check(err, "enqueue sortinitparallelmerge");

	return sortinit_evt;
}

//Setting up the kernel to compute the areas
cl_event intcalc(cl_kernel intcalc_k, cl_command_queue que,
	cl_mem d_v1, cl_mem d_v2, size_t _lws, int nsamples){

	const size_t gws[] = { round_mul_up(nsamples-1, _lws) };	//there are nsamples-1 rectangles
	const size_t lws[] = { _lws };

	cl_event intcalc_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_v1), &d_v1);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, lws[0]*sizeof(float), NULL);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(nsamples), &nsamples);
	ocl_check(err, "set intcalc arg ", i-1);
	
	err = clEnqueueNDRangeKernel(que, intcalc_k, 1, NULL, gws, lws,
		0, NULL, &intcalc_evt);
	ocl_check(err, "enqueue intcalc");

	return intcalc_evt;	
}

//Setting up the kernel to sum all areas
cl_event rect_reduction(cl_kernel reduce4_k, cl_command_queue que,
	cl_mem d_out, cl_mem d_in, cl_int nquarts,
	cl_int lws_, cl_int nwg,
	cl_event init_evt){
	printf("Reduce lws: %d, nwg: %d\n", lws_, nwg);
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

	int nsamples = 128; 
	double leftLimit = 0.0;
	double rightLimit = 1.0;
	if(argc <= 4){
		printf("This program computes the area in an interval under the curve of a function.\n");
		fprintf(stderr, "Use: %s f(x) leftLimit rightLimit nsamples [expectedValue]\n", argv[0]);
		printf("Examples: %s 'sqrt(1-x*x)' 0.0 1.0 1024 0.78539825\n", argv[0]);
		printf("%s 'exp(-x*x)' 0.5 6.3 4096 0.424946\n", argv[0]);
		exit(0);
	}
	else{
		nsamples = atoi(argv[4]);
			if(nsamples & 7){
				nsamples = round_mul_up(nsamples, 8);
				fprintf(stderr, "nsamples must be a multiple of 8. Rounding up to %d\n", nsamples);
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

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit_rect", &err);
	ocl_check(err, "create kernel vecinit");

	cl_kernel intcalc_k = clCreateKernel(prog, "intcalc_randrect_lmem", &err);
	ocl_check(err, "create kernel intcalc");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	cl_kernel sort_k = clCreateKernel(prog, "ParallelMerge_Local", &err);
	ocl_check(err, "create kernel miocountsort");
	
	cl_kernel sort_merge_k = clCreateKernel(prog, "mergebinaryWithRepParallelV4", &err);
	ocl_check(err, "create kernel merging");

	cl_uint seed1, seed2, seed3, seed4;
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	printf("Seeds: %d, %d, %d, %d\n", seed1, seed2, seed3, seed4);

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	if (nsamples/4 < lws){
		nwg = 1;
	}
	else{
		nwg = round_mul_up(nsamples/4, lws)/lws;
		nwg = round_mul_up(nwg, 4);	//Number of work-groups must be a multiple of 4 as well because of the second reduction step
	} 
	printf("lws: %ld, nsamples/4: %d, nwg: %d\n", lws, nsamples/4, nwg);
	size_t memsize = nsamples*sizeof(float);

	cl_mem d_v1, d_v2;

	d_v1 = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL,
		&err);
	ocl_check(err, "create buffer d_v2");

	cl_event vecinit_evt, intcalc_evt, sort_evt, merge_evt1, merge_evt2, reduce_evt[2], read_evt;
	
	vecinit_evt = vecinit(vecinit_k, que, d_v1, leftLimit, rightLimit, seed1, seed2, seed3, seed4, nsamples);
	
	//Sorting time
	sort_evt = sortparallel(sort_k, lws, que, d_v1, nsamples, vecinit_evt);
	int turn=0,pass=1;	
	double total_time_merge=0;
	int current_merge_size = lws;
	merge_evt1=sort_evt;
	while(current_merge_size<nsamples){
		if(turn ==0){
			turn = 1;
			merge_evt2 = sortparallelmerge(sort_merge_k, lws, que, d_v1,d_v2, nsamples, merge_evt1,current_merge_size);
			clWaitForEvents(1, &merge_evt2);

			const double runtime_merge_ms = runtime_ms(merge_evt2);
			total_time_merge += runtime_merge_ms;
			const double merge_bw_gbs = memsize*log2(nsamples)/1.0e6/runtime_merge_ms;
			printf("merge_parziale_lws%i destinazione Sort2: %d float in %gms: %g GB/s %g GE/s\n",
					current_merge_size,nsamples, runtime_merge_ms, merge_bw_gbs, (nsamples)/1.0e6/runtime_merge_ms);
		}
		else{
			turn = 0;
			merge_evt1 = sortparallelmerge(sort_merge_k, lws, que, d_v2,d_v1, nsamples, merge_evt2,current_merge_size);
			clWaitForEvents(1, &merge_evt1);
			const double runtime_merge_ms = runtime_ms(merge_evt1);
			total_time_merge += runtime_merge_ms;
			const double merge_bw_gbs = memsize*log2(nsamples)/1.0e6/runtime_merge_ms;
			printf("merge_parziale_lws%i destinazione Sort1: %d float in %gms: %g GB/s %g GE/s\n",
					current_merge_size,nsamples, runtime_merge_ms, merge_bw_gbs, (nsamples)/1.0e6/runtime_merge_ms);
		}
		current_merge_size<<=1;
		pass++;
	}
	if(turn == 1){	//The sorted random numbers will be in v2, swap pointers
		cl_mem temp = d_v2;
		d_v2 = d_v1;
		d_v1 = temp;
	}
	intcalc_evt = intcalc(intcalc_k, que, d_v1, d_v2, lws, nsamples);

	reduce_evt[0] = rect_reduction(reduce4_k, que, d_v1, d_v2, nsamples/4,
		lws, nwg, intcalc_evt);
	// concludo
	if (nwg > 1) {
		reduce_evt[1] = rect_reduction(reduce4_k, que, d_v1, d_v1, nwg/4,
			lws, 1, reduce_evt[0]);
	} else {
		reduce_evt[1] = reduce_evt[0];
	}

	float sum_result;
	err = clEnqueueReadBuffer(que, d_v1, CL_TRUE, 0, sizeof(sum_result), &sum_result,
		1, reduce_evt + 1, &read_evt);
	ocl_check(err, "read result");

	printf("\n(%s, %s)∫%s dx = %f\n\n", argv[2], argv[3], argv[1], sum_result);
	if (argv[5]) check_accuracy(sum_result, strtod(argv[5], NULL));

	double runtime_init_ms = runtime_ms(vecinit_evt);
	double runtime_sort_ms = runtime_ms(sort_evt) + total_time_merge;
	double runtime_intcalc_ms = runtime_ms(intcalc_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	double init_bw_gbs = memsize/1.0e6/runtime_init_ms;
	double sort_bw_gbs = ((memsize*log2(nsamples)/1.0e6/runtime_sort_ms)+(pass* memsize*log2(nsamples)/1.0e6/total_time_merge))/2;
	double intcalc_bw_gbs = memsize/1.0e6/runtime_intcalc_ms;
	double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("vecinit : %d float in %gms: %g GB/s\n",
		nsamples, runtime_init_ms, init_bw_gbs);
	printf("sort: %d float in %gms: %g GB/s %g GE/s\n",
		nsamples, runtime_sort_ms, sort_bw_gbs, (nsamples)/1.0e6/runtime_sort_ms);
	printf("intcalc : %d float in %gms: %g GB/s\n",
		nsamples, runtime_intcalc_ms, intcalc_bw_gbs);
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[0]);
		const double pass_bw_gbs = (memsize+nwg*sizeof(float))/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			nsamples, runtime_pass_ms, pass_bw_gbs,
			nsamples/1.0e6/runtime_pass_ms);
	}
	if (nwg > 1)
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[1]);
		const double pass_bw_gbs = (nwg*sizeof(float))/1.0e6/runtime_pass_ms;
		printf("reduce1 : %ld float in %gms: %g GB/s %g GE/s\n",
			(lws*nwg), runtime_pass_ms, pass_bw_gbs,
			(lws*nwg)/1.0e6/runtime_pass_ms);
	}
	const double runtime_reduction_ms = total_runtime_ms(reduce_evt[0], reduce_evt[1]);
	const double total_time_ms = total_runtime_ms(vecinit_evt, read_evt);
	printf("reduce : %d float in %gms: %g GE/s\n",
		nsamples, runtime_reduction_ms, nsamples/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);

	clReleaseMemObject(d_v1);
	clReleaseMemObject(d_v2);

	clReleaseKernel(vecinit_k);
	clReleaseKernel(sort_k);
	clReleaseKernel(sort_merge_k);
	clReleaseKernel(intcalc_k);
	clReleaseKernel(reduce4_k);
	clReleaseProgram(prog);
	clReleaseCommandQueue(que);
	clReleaseContext(ctx);
}