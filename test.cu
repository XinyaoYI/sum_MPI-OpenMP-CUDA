/* Compute array sum using MPI, OpenMP and CUDA*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <mpi.h>
#include <omp.h>
#include <stdlib.h>
#include <unistd.h>

#define TOTALN 120120
#define BLOCKS_PerGrid 32
#define THREADS_PerBlock 64 


__global__ void SumArray(float *c, float *a,int m) {
    __shared__ float mycache[THREADS_PerBlock];
    int i = threadIdx.x+blockIdx.x*blockDim.x;
    int j = gridDim.x*blockDim.x;
    int cacheN;
    float sum;
    int k;

    sum=0;

    cacheN=threadIdx.x; 

    while(i<m) {
        sum += a[i];
        i = i+j;
    }

    mycache[cacheN]=sum;
 
    __syncthreads();

    k=THREADS_PerBlock>>1;
    while(k) {
        if(cacheN<k) {
            mycache[cacheN] += mycache[cacheN+k];
        }
        __syncthreads();
        k=k>>1;
    }


    if(cacheN==0) {
        c[blockIdx.x]=mycache[0];
    }
}

 

int main(int argc, char* argv[]) {

    int pid, np, elements_per_process, element_per_GPU;
    float local_sum = 0; 
    MPI_Init(&argc, &argv);
 
    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    elements_per_process = TOTALN / np;

    float a[TOTALN] ;
  
    int j;
    srand48(1<<12);
    for(j=0;j<TOTALN;j++) {
        a[j]=(float) drand48();
    }

    //CPU version
    float sum_serial;
    sum_serial=0;
    for(j=0;j<TOTALN;j++){
        sum_serial += a[j];
    }

    int N = 16; // 16 CPUs and 2GPUs
    float sum_per_thread[N]; 
    for(int i = 0; i < N; i++){sum_per_thread[i] =0;}

    int M =2;
    float c[M*BLOCKS_PerGrid] ;
    element_per_GPU = elements_per_process/(M+1);

    #pragma omp parallel num_threads(N)
    {         

            int tid = omp_get_thread_num();

            if (tid < M) { /* For GPU */
                int GPU_index = (pid*elements_per_process)+tid * element_per_GPU;

                float *dev_a = 0;
                float *dev_c = 0;
               
                cudaMalloc((void**)&dev_c, BLOCKS_PerGrid * sizeof(float));
                cudaMalloc((void**)&dev_a, element_per_GPU * sizeof(float));
                cudaMemcpy(dev_a, &a[GPU_index], element_per_GPU * sizeof(float), cudaMemcpyHostToDevice);
                SumArray<<<BLOCKS_PerGrid, THREADS_PerBlock>>>(dev_c, dev_a, element_per_GPU);

                cudaDeviceSynchronize();
 
                cudaMemcpy(&c[tid*BLOCKS_PerGrid], dev_c, BLOCKS_PerGrid * sizeof(float), cudaMemcpyDeviceToHost);
                
                cudaFree(dev_c);
                cudaFree(dev_a);

                for(int j1=tid*BLOCKS_PerGrid;j1<(tid+1)*BLOCKS_PerGrid;j1++){
                    sum_per_thread[tid] += c[j1];
                    
                }
            }

            else if(tid>=M&&tid !=N-1){
  
                int Nt = (elements_per_process-M*element_per_GPU)/(N-M);

                int i_start = (pid*elements_per_process)+M*element_per_GPU+(tid-M)*Nt;

                int i_end = (pid*elements_per_process)+M*element_per_GPU+((tid-M)+1)*Nt;
   
                for (int i = i_start; i<i_end; i++){
                    sum_per_thread[tid]+=a[i];

                }
            }
            else if (tid==N-1){

                int Nt = (elements_per_process-M*element_per_GPU)-(N-M-1)*((elements_per_process-M*element_per_GPU)/(N-M));
                
                int i_start = (pid+1)*elements_per_process-Nt;
                int i_end = (pid+1)*elements_per_process;

                for (int i = i_start; i<i_end; i++){
                    sum_per_thread[tid]+=a[i];
                }
            }
           
      }

    #pragma omp barrier
    for(int i = 0; i < N; i++){local_sum +=sum_per_thread[i];}

    MPI_Barrier(MPI_COMM_WORLD);
    float global_sum =0;

    MPI_Allreduce ( &local_sum, &global_sum, 1,MPI_FLOAT, MPI_SUM,  MPI_COMM_WORLD );
    MPI_Finalize();

    printf("local_sum=%f; global_sum=%f; sum_serial=%f\n",local_sum,global_sum,sum_serial);

    return 0;
}
