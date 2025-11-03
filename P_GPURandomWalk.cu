// Name: Phil Alcorn	
// GPU random walk. 
// nvcc 16GPURandomWalk.cu -o temp

/*
 What to do:
 This code runs a random walk for 10,000 steps on the CPU.

 1. Use cuRAND to run 20 random walks simultaneously on the GPU, each with a different seed.
    Print out all 20 final positions.

 2. Use cudaMallocManaged(&variable, amount_of_memory_needed);
    This allocates unified memory, which is automatically managed between the CPU and GPU.
    You lose some control over placement, but it saves you from having to manually copy data
    to and from the GPU.
*/

/*
 Purpose:
 To learn how to use cuRAND and unified memory.
*/

/*
 Note:
 The maximum signed int value is 2,147,483,647, so the maximum unsigned int value is 4,294,967,295.

 RAND_MAX is guaranteed to be at least 32,767. When I checked it on my laptop (10/6/2025), it was 2,147,483,647.
 rand() returns a value in [0, RAND_MAX]. It actually generates a list of pseudo-random numbers that depends on the seed.
 This list eventually repeats (this is called its period). The period is usually 2³¹ = 2,147,483,648,
 but it may vary by implementation.

 Because RAND_MAX is odd on this machine and 0 is included, there is no exact middle integer.
 Casting to float as in (float)RAND_MAX / 2.0 divides the range evenly.
 Using integer division (RAND_MAX / 2) would bias results slightly toward the positive side by one value out of 2,147,483,647.

 I know this is splitting hares (sorry, rabbits), but I'm just trying to be as accurate as possible.
 You might do this faster with a clever integer approach, but I’m using floats here for clarity.
*/

#include <stdio.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define NUM_WALKS 20
#define NUM_STEPS 10000

// Each thread performs one random walk
__global__ void gpuRandomWalk(int *xPos, int *yPos, unsigned long long seed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= NUM_WALKS) return;

    // cuRAND state for this thread
	// curandState is a typedef for  curandStateXORWOW_t. There are several
	// different types of state generators available. 
	curandStatePhilox4_32_10_t state;
    curand_init(seed + tid, 0, 0, &state);

    int x = 0, y = 0;

    for (int i = 0; i < NUM_STEPS; i++)
    {
        // curand() returns an unsigned int in [0, 2^32 - 1]
        unsigned int r1 = curand(&state);
        unsigned int r2 = curand(&state);

        // Interpret the LSB as the random direction
		// 1U is just saying "1 as an unsigned int" in the same way 
		// that 1f is just saying "1 as a float".
        x += (r1 & 1U) ? 1 : -1;  // bitwise check for randomness
        y += (r2 & 1U) ? 1 : -1;
    }

    xPos[tid] = x;
    yPos[tid] = y;
}

int main(void)
{
    int *xPos, *yPos;

    // Allocate unified memory
    cudaMallocManaged(&xPos, NUM_WALKS * sizeof(int));
    cudaMallocManaged(&yPos, NUM_WALKS * sizeof(int));
	

	// the c function time(&time_t p) both returns the time and 
	// assigns it to a pointer. Since we only want the time, 
	// we can just pass NULL so it doesn't write anywhere.
    unsigned long long seed = (unsigned long long)time(NULL);

    // Launch kernel (20 threads total)
    dim3 threadsPerBlock(20);
    dim3 numBlocks((NUM_WALKS + threadsPerBlock.x - 1) / threadsPerBlock.x);

    gpuRandomWalk<<<numBlocks, threadsPerBlock>>>(xPos, yPos, seed);
    cudaDeviceSynchronize();

    // Print results
    printf("Final positions after %d steps:\n", NUM_STEPS);
    for (int i = 0; i < NUM_WALKS; i++)
        printf("Walk %2d: (%d, %d)\n", i, xPos[i], yPos[i]);

    // Cleanup
    cudaFree(xPos);
    cudaFree(yPos);
    return 0;
}


