# sum_MPI-OpenMP-CUDA

Compile: nvcc -I/usr/lib/x86_64-linux-gnu/openmpi/include -L/usr/lib/x86_64-linux-gnu/openmpi/lib -lmpi -Xcompiler -fopenmp test.cu -o test
Run: mpirun -np 2 ./test
