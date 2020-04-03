//Definite integral calculator with hyperrectangle rule (stochastic sampling)

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
			if(linectr == 19){
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

void initializeVol(float * h_null, int nvolumes){
	for(int i=0; i<nvolumes; i++){
		h_null[i]=0.0f;
	}
}

//Setting up the kernel to fill a buffer with random numbers
cl_event vecinit(cl_kernel vecinit_k, cl_command_queue que,
	cl_mem d_v, float leftLimit, float rightLimit, cl_uint seed1, cl_uint seed2,
	cl_uint seed3, cl_uint seed4, cl_event * prev_evt, int nsamplesPerDim){

	const size_t gws[] = { nsamplesPerDim/8 };

	cl_event vecinit_evt;
	cl_int err;

	int nwait = 0;
	if(prev_evt) nwait = 1;

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
		nwait, prev_evt, &vecinit_evt);
	ocl_check(err, "enqueue vecinit");

	return vecinit_evt;	
}

cl_event sortparallel(cl_kernel sortinit_k,cl_int _lws, cl_command_queue que,
	cl_mem d_v1, cl_int nels, cl_event * init_event)
{
	const size_t workitem=nels; 
	const size_t gws[] = { round_mul_up(workitem,_lws ) };
	const size_t lws[] = {_lws };
	cl_event sortinit_evt;
	cl_int err;

	int nwait = 0;
	if(init_event) nwait = 1;

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
		nwait, init_event, &sortinit_evt);
	
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

//Setting up the kernel to compute the hypervolumes
cl_event intcalc(cl_kernel intcalc_k, cl_command_queue que,
	cl_mem d_vX, cl_mem d_vY, cl_mem d_vZ, cl_mem d_v2, int nsamplesPerDim){

	const size_t gws[] = { nsamplesPerDim-1, nsamplesPerDim-1, nsamplesPerDim-1 };

	cl_event intcalc_evt;
	cl_int err;

	cl_uint i = 0;
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_vX), &d_vX);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_vY), &d_vY);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_vZ), &d_vZ);
	ocl_check(err, "set intcalc arg ", i-1);
	err = clSetKernelArg(intcalc_k, i++, sizeof(d_v2), &d_v2);
	ocl_check(err, "set intcalc arg ", i-1);
	
	err = clEnqueueNDRangeKernel(que, intcalc_k, 3, NULL, gws, NULL,
		0, NULL, &intcalc_evt);
	ocl_check(err, "enqueue intcalc");

	return intcalc_evt;	
}

//Setting up the kernel to sum all hypervolumes
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

	int nsamplesPerDim = 128; 
	double leftLimitX = 0.0;
	double rightLimitX = 1.0;
	double leftLimitY = 0.0;
	double rightLimitY = 1.0;
	double leftLimitZ = 0.0;
	double rightLimitZ = 1.0;
	if(argc <= 8){
		printf("This program computes the 4D volume enclosed by a cuboid\nunder the curve of a three-variables function\n");
		fprintf(stderr, "Use: %s f(x, y, z) leftLimitX rightLimitX leftLimitY rightLimitY leftLimitZ rightLimitZ nsamplesPerDim [expectedValue]\n", argv[0]);
		printf("Examples: %s 'x*x + y*y + z*z' 0.0 1.0 0.0 1.0 0.0 1.0 16 1\n", argv[0]);
		exit(0);
	}
	else{
		nsamplesPerDim = atoi(argv[8]);
		leftLimitX = strtod(argv[2], NULL);
		rightLimitX = strtod(argv[3], NULL);
		if(nsamplesPerDim & 7){
			nsamplesPerDim = round_mul_up(nsamplesPerDim, 8);
			fprintf(stderr, "The number of samples per dimension must be a multiple of 8. Rounding up to %d\n", nsamplesPerDim);
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

	cl_platform_id p = select_platform();
	cl_device_id d = select_device(p);
	cl_context ctx = create_context(p, d);
	cl_command_queue que = create_queue(ctx, d);
	cl_program prog = create_program("intcalc.ocl", ctx, d);
	cl_int err;

	cl_kernel vecinit_k = clCreateKernel(prog, "vecinit_rect", &err);
	ocl_check(err, "create kernel vecinit");

	cl_kernel intcalc_k = clCreateKernel(prog, "intcalc_randhyperrect", &err);
	ocl_check(err, "create kernel intcalc");

	cl_kernel reduce4_k = clCreateKernel(prog, "reduce4_lmem", &err);
	ocl_check(err, "create kernel reduce4_lmem");

	cl_kernel sort_k = clCreateKernel(prog, "ParallelMerge_Local", &err);
	ocl_check(err, "create kernel miocountsort");
	
	cl_kernel sort_merge_k = clCreateKernel(prog, "mergebinaryWithRepParallelV4", &err);
	ocl_check(err, "create kernel merging");

	//seeds for the edited MWC64X
	cl_uint seed1, seed2, seed3, seed4;
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	printf("Seeds: %d, %d, %d, %d\n", seed1, seed2, seed3, seed4);

	size_t lws;
	err = clGetKernelWorkGroupInfo(reduce4_k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(lws), &lws, NULL);
	ocl_check(err, "Preferred lws multiple for reduce4_k");
	int nwg;
	int nvolumes = (nsamplesPerDim-1)*(nsamplesPerDim-1)*(nsamplesPerDim-1);
	int reduce_volumes = round_mul_up(nvolumes, 4);
	if (reduce_volumes/4 < lws){
		nwg = 1;
		//lws = 32;
	}
	else{
		nwg = round_mul_up(reduce_volumes/4, lws)/lws;
		nwg = round_mul_up(nwg, 4);	//Number of work-groups must be a multiple of 4 as well because of the second reduction step
	} 
	printf("lws: %ld, nvolumes/4: %d, nwg: %d\n", lws, reduce_volumes/4, nwg);
	size_t memsize = nsamplesPerDim*sizeof(float);
	size_t vol_memsize = reduce_volumes*sizeof(float);

	cl_mem d_vX, d_vY, d_vZ;
	cl_mem d_v1, d_v2;

	d_vX = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_vX");
	d_vY = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_vY");
	d_vZ = clCreateBuffer(ctx, 
		CL_MEM_READ_WRITE | CL_MEM_HOST_READ_ONLY,
		memsize, NULL, &err);
	ocl_check(err, "create buffer d_vZ");
	d_v1 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE,
		vol_memsize, NULL,
		&err);
	ocl_check(err, "create buffer d_v1");
	d_v2 = clCreateBuffer(ctx,
		CL_MEM_READ_WRITE,
		vol_memsize, NULL,
		&err);
	ocl_check(err, "create buffer d_v2");
	
	float *h_null = (float*) malloc(vol_memsize);
	initializeVol(h_null, reduce_volumes);

	err = clEnqueueWriteBuffer(que, d_v1, CL_TRUE, 0, vol_memsize, h_null, 0, NULL, NULL);
	ocl_check(err, "initialize volumes buffer v1");

	err = clEnqueueWriteBuffer(que, d_v2, CL_TRUE, 0, vol_memsize, h_null, 0, NULL, NULL);
	ocl_check(err, "initialize volumes buffer v2");

	free((void*)h_null);

	cl_event vecinitX_evt, vecinitY_evt, vecinitZ_evt;
	cl_event sortX_evt, sortY_evt, sortZ_evt;
	cl_event mergeX_evt1, mergeX_evt2, mergeY_evt1, mergeY_evt2, mergeZ_evt1, mergeZ_evt2;
	cl_event reduce_evt[2], read_evt;
	cl_event intcalc_evt;
	
	vecinitX_evt = vecinit(vecinit_k, que, d_vX, leftLimitX, rightLimitX, seed1, seed2, seed3, seed4, NULL, nsamplesPerDim);
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	vecinitY_evt = vecinit(vecinit_k, que, d_vY, leftLimitY, rightLimitY, seed1, seed2, seed3, seed4, &vecinitX_evt, nsamplesPerDim);
	initializeSeeds(&seed1, &seed2, &seed3, &seed4);
	vecinitZ_evt = vecinit(vecinit_k, que, d_vZ, leftLimitZ, rightLimitZ, seed1, seed2, seed3, seed4, &vecinitY_evt, nsamplesPerDim);

	//Sorting time X
	sortX_evt = sortparallel(sort_k, lws, que, d_vX, nsamplesPerDim, &vecinitZ_evt);
	int turn=0,pass=1;	
	double total_time_mergeX=0;
	int current_merge_size = lws;
	mergeX_evt1=sortX_evt;
	while(current_merge_size<nsamplesPerDim){
		if(turn ==0){
			turn = 1;
			mergeX_evt2 = sortparallelmerge(sort_merge_k, lws, que, d_vX,d_v2, nsamplesPerDim, mergeX_evt1,current_merge_size);
			clWaitForEvents(1, &mergeX_evt2);

			const double runtime_merge_ms = runtime_ms(mergeX_evt2);
			total_time_mergeX += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort2: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		else{
			turn = 0;
			mergeX_evt1 = sortparallelmerge(sort_merge_k, lws, que, d_v2,d_vX, nsamplesPerDim, mergeX_evt2,current_merge_size);
			clWaitForEvents(1, &mergeX_evt1);
			const double runtime_merge_ms = runtime_ms(mergeX_evt1);
			total_time_mergeX += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort1: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		current_merge_size<<=1;
		pass++;
	}
	if(turn == 1){	//The sorted random numbers will be in v2, swap pointers
		cl_mem temp = d_v2;
		d_v2 = d_vX;
		d_vX = temp;
	}
	turn = 0, pass=1;

	//Sorting time Y
	sortY_evt = sortparallel(sort_k, lws, que, d_vY, nsamplesPerDim, NULL);
	double total_time_mergeY=0;
	current_merge_size = lws;
	mergeY_evt1=sortY_evt;
	while(current_merge_size<nsamplesPerDim){
		if(turn ==0){
			turn = 1;
			mergeY_evt2 = sortparallelmerge(sort_merge_k, lws, que, d_vY,d_v2, nsamplesPerDim, mergeY_evt1,current_merge_size);
			clWaitForEvents(1, &mergeY_evt2);

			const double runtime_merge_ms = runtime_ms(mergeY_evt2);
			total_time_mergeY += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort2: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		else{
			turn = 0;
			mergeY_evt1 = sortparallelmerge(sort_merge_k, lws, que, d_v2,d_vY, nsamplesPerDim, mergeY_evt2,current_merge_size);
			clWaitForEvents(1, &mergeY_evt1);
			const double runtime_merge_ms = runtime_ms(mergeY_evt1);
			total_time_mergeY += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort1: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		current_merge_size<<=1;
		pass++;
	}
	if(turn == 1){	//The sorted random numbers will be in v2, swap pointers
		cl_mem temp = d_v2;
		d_v2 = d_vY;
		d_vY = temp;
	}

	turn = 0, pass = 0;
	//Sorting time Z
	sortZ_evt = sortparallel(sort_k, lws, que, d_vZ, nsamplesPerDim, NULL);	
	double total_time_mergeZ=0;
	current_merge_size = lws;
	mergeZ_evt1=sortZ_evt;
	while(current_merge_size<nsamplesPerDim){
		if(turn ==0){
			turn = 1;
			mergeZ_evt2 = sortparallelmerge(sort_merge_k, lws, que, d_vZ, d_v2, nsamplesPerDim, mergeZ_evt1,current_merge_size);
			clWaitForEvents(1, &mergeZ_evt2);

			const double runtime_merge_ms = runtime_ms(mergeZ_evt2);
			total_time_mergeZ += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort2: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		else{
			turn = 0;
			mergeZ_evt1 = sortparallelmerge(sort_merge_k, lws, que, d_v2,d_vZ, nsamplesPerDim, mergeZ_evt2,current_merge_size);
			clWaitForEvents(1, &mergeZ_evt1);
			const double runtime_merge_ms = runtime_ms(mergeZ_evt1);
			total_time_mergeZ += runtime_merge_ms;
			//const double merge_bw_gbs = memsize*log2(nsamplesPerDim)/1.0e6/runtime_merge_ms;
			//printf("merge_parziale_lws%i destinazione Sort1: %d int in %gms: %g GB/s %g GE/s\n",
			//		current_merge_size,nsamplesPerDim, runtime_merge_ms, merge_bw_gbs, (nsamplesPerDim)/1.0e6/runtime_merge_ms);
		}
		current_merge_size<<=1;
		pass++;
	}
	if(turn == 1){	//The sorted random numbers will be in v2, swap pointers
		cl_mem temp = d_v2;
		d_v2 = d_vZ;
		d_vZ = temp;
	}

	intcalc_evt = intcalc(intcalc_k, que, d_vX, d_vY, d_vZ, d_v2, nsamplesPerDim);

	reduce_evt[0] = rect_reduction(reduce4_k, que, d_v1, d_v2, reduce_volumes/4,
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

	printf("\n(%s, %s)∫(%s, %s)∫(%s, %s)∫[%s] dxdydz = %g\n\n", argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[1], sum_result);
	if (argv[9]) check_accuracy(sum_result, strtod(argv[9], NULL));

	double runtime_init_ms = runtime_ms(vecinitX_evt) + runtime_ms(vecinitY_evt) + runtime_ms(vecinitZ_evt);
	double runtime_sort_ms = runtime_ms(sortX_evt) + runtime_ms(sortY_evt) + runtime_ms(sortZ_evt) + total_time_mergeX + total_time_mergeY + total_time_mergeZ;
	double runtime_intcalc_ms = runtime_ms(intcalc_evt);
	double runtime_read_ms = runtime_ms(read_evt);

	double init_bw_gbs = 3*memsize/1.0e6/runtime_init_ms;
	double sort_bw_gbs = ((3*memsize*log2(nsamplesPerDim)/1.0e6/runtime_sort_ms)+(pass* 3*memsize*log2(nsamplesPerDim)/1.0e6/(total_time_mergeX + total_time_mergeY + total_time_mergeZ)))/2;
	double intcalc_bw_gbs = vol_memsize/1.0e6/runtime_intcalc_ms;
	double read_bw_gbs = sizeof(float)/1.0e6/runtime_read_ms;

	printf("vecinit : %d float in %gms: %g GB/s\n",
		nsamplesPerDim, runtime_init_ms, init_bw_gbs);
	printf("sort: %d float in %gms: %g GB/s %g GE/s\n",
		3*nsamplesPerDim, runtime_sort_ms, sort_bw_gbs, (3*nsamplesPerDim)/1.0e6/runtime_sort_ms);
	printf("intcalc : %d float in %gms: %g GB/s\n",
		nvolumes, runtime_intcalc_ms, intcalc_bw_gbs);
	{
		const double runtime_pass_ms = runtime_ms(reduce_evt[0]);
		const double pass_bw_gbs = (vol_memsize+nwg*sizeof(float))/1.0e6/runtime_pass_ms;
		printf("reduce0 : %d float in %gms: %g GB/s %g GE/s\n",
			reduce_volumes, runtime_pass_ms, pass_bw_gbs,
			reduce_volumes/1.0e6/runtime_pass_ms);
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
	const double total_time_ms = total_runtime_ms(vecinitX_evt, read_evt);
	printf("reduce : %d float in %gms: %g GE/s\n",
		reduce_volumes, runtime_reduction_ms, reduce_volumes/1.0e6/runtime_reduction_ms);
	printf("read : sum (float) in %gms: %g GB/s\n",
		runtime_read_ms, read_bw_gbs);
	printf("\nTotal time: %g ms.\n", total_time_ms);

	clReleaseMemObject(d_vX);
	clReleaseMemObject(d_vY);
	clReleaseMemObject(d_vZ);
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