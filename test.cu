#include <mpi.h> 
#include <stdio.h> 
#include <stdlib.h> 
#include <unistd.h> 
#include <omp.h>
#include <math.h>

// size of array 
#define n 1000 

__global__ void vecAdd(float *a,float sum,int m)
{
    // Get our global thread ID
    int id = blockIdx.x*blockDim.x+threadIdx.x;
 
    // Make sure we do not go out of bounds
    if (id < n)
        sum += a[id];
    //printf("cuda");
}

float a[1000];
int i;

//float a[10000] = { 0, 2, 3, 4, 5, 6, 7, 8, 9, 10 }; 

// Temporary array for slave process 
float a2[1000]; 



int main(int argc, char* argv[]) 
{ 
for(i = 0; i < n; i++) {
    a[i] = (float) drand48();
 }
	int pid, np, 
		elements_per_process, 
		n_elements_recieved; 
	// np -> no. of processes 
	// pid -> process id 

	MPI_Status status; 

	// Creation of parallel processes 
	MPI_Init(&argc, &argv); 

	// find out process ID, 
	// and how many processes were started 
	MPI_Comm_rank(MPI_COMM_WORLD, &pid); 
	MPI_Comm_size(MPI_COMM_WORLD, &np); 

	// master process 
	if (pid == 0) { 
		int index, i; 
		elements_per_process = n / np; 

		// check if more than 1 processes are run 
		if (np > 1) { 
			// distributes the portion of array 
			// to child processes to calculate 
			// their partial sums 
			for (i = 1; i < np - 1; i++) { 
				index = i * elements_per_process; 

				MPI_Send(&elements_per_process, 
						1, MPI_INT, i, 0, 
						MPI_COMM_WORLD); 
				MPI_Send(&a[index], 
						elements_per_process, 
						MPI_INT, i, 0, 
						MPI_COMM_WORLD); 
			} 

			// last process adds remaining elements 
			index = i * elements_per_process; 
			int elements_left = n - index; 

			MPI_Send(&elements_left, 
					1, MPI_INT, 
					i, 0, 
					MPI_COMM_WORLD); 
			MPI_Send(&a[index], 
					elements_left, 
					MPI_INT, i, 0, 
					MPI_COMM_WORLD); 
		} 

		// master process add its own sub array 
		float h_sum = 0;
                float *d_sum;
                float sum = 0;

 
                #pragma omp parallel for
		for (i = 0; i < 4; i++) 
		    {
                        float *d_a;
                        cudaMalloc(&d_a, (elements_per_process/4)*sizeof(float));
                        cudaMalloc(&d_sum, sizeof(float));
                        //an error could be here
                        cudaMemcpy( d_a, &a[i*(elements_per_process/4)], (elements_per_process/4)*sizeof(float), cudaMemcpyHostToDevice);
                        cudaMemcpy( d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice);
                        int blockSize, gridSize;
                        blockSize = 1024;
                        gridSize = (int)ceil((float)(elements_per_process/4)/blockSize);
                        vecAdd<<<gridSize, blockSize>>>(d_a, *d_sum, (elements_per_process/4));
                        cudaMemcpy( &h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost );
                        sum += h_sum; printf("Sum of array is : %f\n", sum); 
                        cudaFree(d_a);
                        cudaFree(d_sum);
                       printf("asd");
                    }
                
                      
		// collects partial sums from other processes 
		int tmp; 
		for (i = 1; i < np; i++) { 
			MPI_Recv(&tmp, 1, MPI_INT, 
					MPI_ANY_SOURCE, 0, 
					MPI_COMM_WORLD, 
					&status); 
			int sender = status.MPI_SOURCE; 

			sum += tmp; 
		} 

		// prints the final sum of array 
		printf("Sum of array isasdfhjk : %f\n", sum); 
	} 
	// slave processes 
	else { 
		MPI_Recv(&n_elements_recieved, 
				1, MPI_INT, 0, 0, 
				MPI_COMM_WORLD, 
				&status); 

		// stores the received array segment 
		// in local array a2 
		MPI_Recv(&a2, n_elements_recieved, 
				MPI_INT, 0, 0, 
				MPI_COMM_WORLD, 
				&status); 

		// calculates its partial sum 
		int partial_sum = 0; 
		for (int i = 0; i < n_elements_recieved; i++) 
			partial_sum += a2[i]; 

		// sends the partial sum to the root process 
		MPI_Send(&partial_sum, 1, MPI_INT, 
				0, 0, MPI_COMM_WORLD); 
	} 

	// cleans up all MPI state before exit of process 
	MPI_Finalize(); 

	return 0; 
} 
