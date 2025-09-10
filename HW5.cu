// Name: Phil Alcorn
// Device query
// nvcc HW5.cu -o temp
/*
 What to do:
 This code prints out useful information about the GPU(s) in your machine, 
 but there is much more data available in the cudaDeviceProp structure.

 Extend this code so that it prints out all the information 
 about the GPU(s) in your system. 

 Also, and this is the fun part, be prepared to explain what each 
 piece of information means. 
*/

// Include files
#include <stdio.h>

// Defines (ie, MACRO abuse)
# define PROP_STRUCT prop
# define PRINT_(text,field, format) \
	printf("%s: %" #format "\n", #text,  PROP_STRUCT.field)

// Global variables (bad)

// Function prototypes

void cudaErrorCheck(const char*, int);
int printDeviceCount();
void printStats(int n);
void printUUID(const char uuid[16]);



// Program begins here
int main()
{
	printf("\n");
	int count = printDeviceCount();

	for (int i = 0; i < count; i++) 
	{
		printStats(i);
	}

	printf("\n");	
	return(0);
}


void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, File = %s, Line = %d\n", cudaGetErrorString(error), file, line);
		exit(0);
	}
}


int printDeviceCount()
{
	int count;
	cudaGetDeviceCount(&count);
	cudaErrorCheck(__FILE__, __LINE__);

	if (count>1)
	{
		printf("You have %d GPUs in this machine.\n\n", count);
	} 
	else
	{
		printf("You have %d GPU in this machine.\n\n", count);
	}

	return count;
}


void printStats(int n)
{
	cudaDeviceProp PROP_STRUCT;
	cudaErrorCheck(__FILE__, __LINE__);

	cudaGetDeviceProperties(&PROP_STRUCT, n);
	cudaErrorCheck(__FILE__, __LINE__);

	printf("--- General Information for device %d ---\n", n);
	PRINT_(Device Name, name, s);
	PRINT_(Major Version, major, d);
	PRINT_(Minor Version,minor, d);
	PRINT_(Clock Rate, clockRate, d);
	PRINT_(Device OVerlap Enabled, deviceOverlap, d);
	PRINT_(Kernel Execution Timeout Enabled,
			kernelExecTimeoutEnabled, d);	
	printf("\n\n");

	printf("--- Memory Information for device %d ---\n", n);
	PRINT_(Total Global Memory, totalGlobalMem, ld);
	PRINT_(Total Constant Memory, totalConstMem, ld);
	PRINT_(Max Memory Pitch, memPitch, ld);
	PRINT_(Texture Alignment,textureAlignment, ld); 
	printf("\n\n");

	printf("--- MP Information for device %d ---\n", n);
	PRINT_(Multiprocessor Count, multiProcessorCount, d);
	PRINT_(Shared Memory Per MP, sharedMemPerBlock, d);
	PRINT_(Registers Per MP, regsPerBlock, ld);
	PRINT_(Threads in Warp, warpSize, d);
	PRINT_(Max Threads per Block, maxThreadsPerBlock, d);
	
	printf("Max thread dimensions: (%d, %d, %d)\n", prop.maxThreadsDim[0],
			prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

	printf("Max grid dimensions: (%d, %d, %d)\n", prop.maxGridSize[0], 
			prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("\n\n");

	printf("--- All Variables ----\n");
	PRINT_(ECC Enabled, ECCEnabled, d);
	PRINT_(Access Policy Max Window Size, accessPolicyMaxWindowSize, d);
	PRINT_(Async Engine Count, asyncEngineCount, d);
	PRINT_(Can Map Host Memory, canMapHostMemory, d);
	PRINT_(Can Use Host Pointer For Registered Mem, 
			canUseHostPointerForRegisteredMem, d);
	PRINT_(Cluster Launch, clusterLaunch, d);
	PRINT_(Compute Preemption Supported, computePreemptionSupported, d);
	PRINT_(Concurrent Kernels, concurrentKernels, d);
	PRINT_(Concurrent Managed Access, concurrentManagedAccess, d);
	PRINT_(Cooperative Launch, cooperativeLaunch, d);
	PRINT_(Deferred Mapping CUDA Array Supported, 
			deferredMappingCudaArraySupported, d);
	//PRINT_(Device NUMA Config, deviceNumaConfig, d);
	//PRINT_(Device NUMA Id, deviceNumaId, d);
	PRINT_(Direct Managed Mem Access From Host, directManagedMemAccessFromHost, d);
	PRINT_(Global L1 Cache Supported, globalL1CacheSupported, d);
	PRINT_(GPUDirect RDMA Flush Writes Options, 
			gpuDirectRDMAFlushWritesOptions, u);  // unsigned int
	PRINT_(GPUDirect RDMA Supported, gpuDirectRDMASupported, d);
	PRINT_(GPUDirect RDMA Writes Ordering, gpuDirectRDMAWritesOrdering, d);
	//PRINT_(GPU PCI Device ID, gpuPciDeviceID, u);  // unsigned int
	//PRINT_(GPU PCI Subsystem ID, gpuPciSubsystemID, u);
	PRINT_(Host Native Atomic Supported, hostNativeAtomicSupported, d);
	//PRINT_(Host NUMA Id, hostNumaId, d);
	//PRINT_(Host NUMA Multinode IPC Supported, hostNumaMultinodeIpcSupported, d);
	PRINT_(Host Register Read-Only Supported, hostRegisterReadOnlySupported, d);
	PRINT_(Host Register Supported, hostRegisterSupported, d);
	PRINT_(Integrated, integrated, d);
	PRINT_(IPC Event Supported, ipcEventSupported, d);
	PRINT_(Is Multi-GPU Board, isMultiGpuBoard, d);
	PRINT_(L2 Cache Size, l2CacheSize, d);
	PRINT_(Local L1 Cache Supported, localL1CacheSupported, d);
	PRINT_(LUID, luid, s);  // 8-byte array, treat as string or hex
	PRINT_(LUID Device Node Mask, luidDeviceNodeMask, u);
	PRINT_(Compute Capability Major, major, d);
	PRINT_(Managed Memory, managedMemory, d);
	PRINT_(Max Blocks Per Multi-Processor, maxBlocksPerMultiProcessor, d);
	PRINT_(Max Grid Size [0], maxGridSize[0], d);
	PRINT_(Max Grid Size [1], maxGridSize[1], d);
	PRINT_(Max Grid Size [2], maxGridSize[2], d);
	PRINT_(Max Surface 1D, maxSurface1D, d);
	PRINT_(Max Surface 1D Layered [0], maxSurface1DLayered[0], d);
	PRINT_(Max Surface 1D Layered [1], maxSurface1DLayered[1], d);
	PRINT_(Max Surface 2D [0], maxSurface2D[0], d);
	PRINT_(Max Surface 2D [1], maxSurface2D[1], d);
	PRINT_(Max Surface 2D Layered [0], maxSurface2DLayered[0], d);
	PRINT_(Max Surface 2D Layered [1], maxSurface2DLayered[1], d);
	PRINT_(Max Surface 2D Layered [2], maxSurface2DLayered[2], d);
	PRINT_(Max Surface 3D [0], maxSurface3D[0], d);
	PRINT_(Max Surface 3D [1], maxSurface3D[1], d);
	PRINT_(Max Surface 3D [2], maxSurface3D[2], d);
	PRINT_(Max Surface Cubemap, maxSurfaceCubemap, d);
	PRINT_(Max Surface Cubemap Layered [0], maxSurfaceCubemapLayered[0], d);
	PRINT_(Max Surface Cubemap Layered [1], maxSurfaceCubemapLayered[1], d);
	PRINT_(Max Texture 1D, maxTexture1D, d);
	PRINT_(Max Texture 1D Layered [0], maxTexture1DLayered[0], d);
	PRINT_(Max Texture 1D Layered [1], maxTexture1DLayered[1], d);
	PRINT_(Max Texture 1D Mipmap, maxTexture1DMipmap, d);
	PRINT_(Max Texture 2D [0], maxTexture2D[0], d);
	PRINT_(Max Texture 2DGather [0], maxTexture2DGather[0], d);
	PRINT_(Max Texture 2DGather [1], maxTexture2DGather[1], d);
	PRINT_(Max Texture 2D Layered [0], maxTexture2DLayered[0], d);
	PRINT_(Max Texture 2D Layered [1], maxTexture2DLayered[1], d);
	PRINT_(Max Texture 2D Layered [2], maxTexture2DLayered[2], d);
	PRINT_(Max Texture 2D Linear [0], maxTexture2DLinear[0], d);
	PRINT_(Max Texture 2D Linear [1], maxTexture2DLinear[1], d);
	PRINT_(Max Texture 2D Linear [2], maxTexture2DLinear[2], d);
	PRINT_(Max Texture 2D Mipmap [0], maxTexture2DMipmap[0], d);
	PRINT_(Max Texture 2D Mipmap [1], maxTexture2DMipmap[1], d);
	PRINT_(Max Texture 3D [0], maxTexture3D[0], d);
	PRINT_(Max Texture 3D [1], maxTexture3D[1], d);
	PRINT_(Max Texture 3D [2], maxTexture3D[2], d);
	PRINT_(Max Texture 3D Alt [0], maxTexture3DAlt[0], d);
	PRINT_(Max Texture 3D Alt [1], maxTexture3DAlt[1], d);
	PRINT_(Max Texture 3D Alt [2], maxTexture3DAlt[2], d);
	PRINT_(Max Texture Cubemap, maxTextureCubemap, d);
	PRINT_(Max Texture Cubemap Layered [0], maxTextureCubemapLayered[0], d);
	PRINT_(Max Texture Cubemap Layered [1], maxTextureCubemapLayered[1], d);
	PRINT_(Max Threads Dim [0], maxThreadsDim[0], d);
	PRINT_(Max Threads Dim [1], maxThreadsDim[1], d);
	PRINT_(Max Threads Dim [2], maxThreadsDim[2], d);
	PRINT_(Max Threads Per Block, maxThreadsPerBlock, d);
	PRINT_(Max Threads Per Multi-Processor, maxThreadsPerMultiProcessor, d);
	PRINT_(Memory Pitch, memPitch, zu);
	PRINT_(Memory Bus Width, memoryBusWidth, d);
	PRINT_(Memory Pool Supported Handle Types, memoryPoolSupportedHandleTypes, u);
	PRINT_(Memory Pools Supported, memoryPoolsSupported, d);
	PRINT_(Compute Capability Minor, minor, d);
	//PRINT_(MPS Enabled, mpsEnabled, d);
	PRINT_(Multi-GPU Board Group ID, multiGpuBoardGroupID, d);
	PRINT_(Multi-Processor Count, multiProcessorCount, d);
	PRINT_(Device Name, name, s);
	PRINT_(Pageable Memory Access, pageableMemoryAccess, d);
	PRINT_(Pageable Memory Access Uses Host Page Tables, 
			pageableMemoryAccessUsesHostPageTables, d);
	PRINT_(PCI Bus ID, pciBusID, d);
	PRINT_(PCI Device ID, pciDeviceID, d);
	PRINT_(PCI Domain ID, pciDomainID, d);
	PRINT_(Persisting L2 Cache Max Size, persistingL2CacheMaxSize, d);
	PRINT_(Registers Per Block, regsPerBlock, d);
	PRINT_(Registers Per Multi-Processor, regsPerMultiprocessor, d);
	PRINT_(Reserved [0], reserved[0], d);
	// (…continue for reserved[1] through reserved[55] if desired)
	PRINT_(Reserved Shared Mem Per Block, reservedSharedMemPerBlock, zu);
	PRINT_(Shared Memory Per Block, sharedMemPerBlock, zu);
	PRINT_(Shared Memory Per Block Opt-In, sharedMemPerBlockOptin, zu);
	PRINT_(Shared Memory Per Multi-Processor, sharedMemPerMultiprocessor, zu);
	PRINT_(Sparse CUDA Array Supported, sparseCudaArraySupported, d);
	PRINT_(Stream Priorities Supported, streamPrioritiesSupported, d);
	PRINT_(Surface Alignment, surfaceAlignment, zu);
	PRINT_(TCC Driver, tccDriver, d);
	PRINT_(Texture Alignment, textureAlignment, zu);
	PRINT_(Texture Pitch Alignment, texturePitchAlignment, zu);
	PRINT_(Timeline Semaphore Interop Supported, 
			timelineSemaphoreInteropSupported, d);
	PRINT_(Total Constant Memory, totalConstMem, zu);
	PRINT_(Total Global Memory, totalGlobalMem, zu);
	PRINT_(Unified Addressing, unifiedAddressing, d);
	PRINT_(Unified Function Pointers, unifiedFunctionPointers, ld);

	//PRINT_(UUID, uuid, s);  // 16-byte identifier, print as hex or similar
	printf("UUID");
	printUUID(PROP_STRUCT.uuid.bytes);

	PRINT_(Warp Size, warpSize, d);
}


void printUUID(const char uuid[16]) {
    for (int i = 0; i < 16; i++) {
        printf("%02x", static_cast<unsigned char>(uuid[i]));
        // Optionally insert dashes to match UUID convention
        if (i == 3 || i == 5 || i == 7 || i == 9) {
            printf("-");
        }
    }
    printf("\n");
}
