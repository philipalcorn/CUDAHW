/****IMPORTANT*****/
//To Run: ./temp {width} {height}

// Name: Phil Alcorn
// Simple Julia CPU.
// nvcc HW7.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.

/*
 What to do:
 This code displays a simple Julia set fractal using the GPU.
 But it only runs on a window of 1024X1024.
 Extend it so that it can run on any given window size.
 Also, color it to your liking. I will judge you on your artisct flare. 
 Don't cute off your ear or anything but make Vincent wish he had, had a GPU.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>
#include <sys/time.h>

// Defines
#define CHECK() cudaErrorCheck(__FILE__, __LINE__)

//globals- generally bad, but we need this one for GLUT
float* pixels_CPU;
float* pixels_GPU;
dim3 blockSize;
dim3 gridSize;
long long start_time;

// Still "constant" but they are assigned at program start
int WIDTH;
int HEIGHT;

// Constants
// const is better than define as it forces type checking
const float TIME_SCALE = 0.001;
const float X_LOWER_BOUND = -2; 
const float X_UPPER_BOUND =  2;
const float Y_LOWER_BOUND = -2;
const float Y_UPPER_BOUND = 2;
const float A =  -0.824;			//Real part of C
const float B  = -0.1711;			//Imaginary part of C Global variables unsigned int 
const int MAX_MAGNITUDE = 20;
const int MAX_ITERATIONS = 200;		// considered not escaped if you make it this far
const int THREAD_WIDTH = 32;		// Square Root of 24
const int THREAD_HEIGHT = 32;


// Function prototypes
void cudaErrorCheck(const char*, int);
void checkAllocation(void* ptr);

void setUpDevices(dim3* blockSize, dim3* gridSize);
void checkSetUpDevices(dim3* b, dim3* g);

long long get_current_time_ms() ;

// input either x or y pixel coordinates and get corresponding number
// based on size of screen
__device__ float getNumFromPX(int position_px, 
							  int length_dim, 
							  float real_min, 
							  float real_max);
__global__ void checkGetNumFromPX(); 

// x, y, channel (0 = red, 1 = green, 2 = blue);
__device__ int getPixelIndex(int x, int y, int channel, int width);
__global__ void checkGetPixelIndex();

__device__ float getMagnitude(float x, float y);
__global__ void checkGetMagnitude();


//used to progress each iteration of the fractal algorithm
__device__ void step(float* re, float* im, float A, float B);

__device__ float getAnimationValue(float min, float max, float timestep);

// Takes as input the memory address of your GPU pixel array
__global__ void checkEscapeGPU(int width, int height, float* pixel_array, float time);
void display();
void timer(int value);


int main(int argc, char** argv)
{
	//argc is arg count, argv is arg vector
	if(argc !=3) 
	{
		printf("Run the program like this: ./<filename> <width> <height>\n");
		exit(1);
	}
	WIDTH = atoi(argv[1]);
	HEIGHT = atoi(argv[2]);

	// allocate space for our RGB values
	pixels_CPU = (float*)malloc(WIDTH*HEIGHT*3*sizeof(float)); 
	checkAllocation(pixels_CPU);

	// We can fill this array on the GPU and then transfer it to the CPU to display
	cudaMalloc(&pixels_GPU, WIDTH*HEIGHT*3*sizeof(float));
	CHECK();

	setUpDevices(&blockSize, &gridSize);
	CHECK();
	
	

	/*
	long long time_elapsed = get_current_time_ms() - start_time;
	checkEscapeGPU<<<gridSize, blockSize>>>(WIDTH, HEIGHT, pixels_GPU, time_elapsed);
	CHECK();
	cudaDeviceSynchronize();
	*/
	
	cudaMemcpy(pixels_CPU, pixels_GPU,WIDTH*HEIGHT*3*sizeof(float), cudaMemcpyDeviceToHost);
	CHECK();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Fractals--Man--Fractals");
	glutDisplayFunc(display);

	start_time = get_current_time_ms();
	glutTimerFunc(16,timer,0);

	glutMainLoop();
	//Here are a variety of checks to run to verify functionality
	
	/*
	 
	//Checking our block size setup
	checkSetUpDevices(&blockSize, &gridSize);
	CHECK();

	// Checking our conversion from px coordinate to real number value 
	checkGetNumFromPX<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize(); // need to call this or printf's might not be sent to host
	CHECK();
	
	// Checking our pixel indexing funciton:
	checkGetPixelIndex<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize();
	CHECK();
	
	// Checking the magnitude function:
	checkGetMagnitude<<<1,1>>>(); 
	CHECK();
	cudaDeviceSynchronize();
	CHECK();
	
	*/
	
	free(pixels_CPU);
	cudaFree(pixels_GPU);
}

// Unit Testing
void cudaErrorCheck(const char *file, int line)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: message = %s, \
				File = %s, Line = %d\n", cudaGetErrorString(error), 
				file, line);
		exit(0);
	}
}


// Unit Testing
void checkAllocation(void* ptr) 
{
	if (!ptr) 
	{
		printf("Memory Allocation failed.\n");
		exit(1);
	}
}


void setUpDevices(dim3* blockSize, dim3* gridSize) 
{
	// One thread per pixel in a 1024x1024 grid, anything larger 
	// needs more blocks 
	blockSize->x = THREAD_WIDTH;
	blockSize->y = THREAD_HEIGHT;
	blockSize->z = 1;

	//The minimum number of blocks to hold the pixels in the x and y directions	
	gridSize->x = (WIDTH-1)/blockSize->x + 1; 
	gridSize->y = (HEIGHT-1)/blockSize->y + 1; 
	gridSize->z = 1; // only two dimensions
}


// Unit Testing
void checkSetUpDevices(dim3* b, dim3* g) 
{
		
	printf("---Checking Device Setup---\n");
	printf("Block Size: %d, %d, %d\n"
			"Grid Size: %d, %d, %d\n",
			b->x, b->y, b->z,
			g->x, g->y, g->z);
	printf("\n");
}


long long get_current_time_ms() 
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (long long)(tv.tv_sec)*1000 + (tv.tv_usec / 1000);
}


// Maps a pixel value to a real number. 
// The pixel position is either the x or y coordinate, and then the 
// Other three parameters are relevant to that dimension only (all x or all y)
__device__ float getNumFromPX(int position_px, 
							 int length_dim, 
							 float real_min, 
							 float real_max)
{
	return real_min + (position_px * (real_max - real_min)/(length_dim-1));
} 


// Unit Testing
__global__ void checkGetNumFromPX() 
{
	float temp1 = getNumFromPX(0,	1024,	 0,	1);
	float temp2 = getNumFromPX(1023,	1024,	 0,	1);
	float temp3 = getNumFromPX(0,	4096,	-2,	1);
	float temp4 = getNumFromPX(4096,	4096,	-2,	1);
	printf("---Checking PX Coordinate Mapping---\n");
	printf("X: mapping from 0 to -1, 1024 pixels wide\n"
			"Pixel pos 0: %f\n" 
			"Pixel pos 1023: %f\n", 
			temp1, temp2);
	printf("Y: mapping from -2 to 1, 4096 pixels tall\n"
			"Pixel pos 0: %f\n"
			"Pixel pos 1023: %f\n", 
			temp3, temp4);
	printf(" (+/- some floating point error)\n");
	printf("\n");
}


//Given an X,Y pixel, returns the position in the single dimension RGB array
// (which is three times as long as the corresponding single dimension 
// pixel array)
__device__ int getPixelIndex(int x, int y, int channel, int width) 
{
	
	return (y * width + x) * 3 + channel;
}


// Unit Testing
__global__ void checkGetPixelIndex(int width)
{
	printf("---Checking PX Index Conversion---\n");
	printf("Pixel (0, 0, Blue(2))\n");
	printf("Index: %d\n", getPixelIndex(0,0,2, width));
	printf("Pixel (100, 2, Green(1))\n");
	printf("Index: %d\n", getPixelIndex(100,2,1, width));
	printf("If width is 1024x1024, this coordinate should be: \n\n"
			 "\t(y-coord*WIDTH) *3 + x-coord*3 + color_offset \n"
			 "or, \n"
			 "\t2*1024*3 + 100*3 + 1 \n" 
			 "\twhich equals 6,445\n");
	printf("\n");
}


__device__ float getMagnitude(float x, float y) 
{
	return sqrt(x*x + y*y);
}


// Unit Testing
__global__ void checkGetMagnitude() 
{
	printf("Mag of (3,4): %f\n", getMagnitude(3,4));
}


// Note: This function modifies the original input.
// Applies a single iteration of our fractal algorithm to the input. 
__device__ void step(float* re, float* im, float A, float B) 
{
	float re_initial = *re;

	*re = re_initial*re_initial - (*im)*(*im) + A;
	*im = 2 * re_initial * (*im) + B; 
}

// "lingering" sine wave
// https://math.stackexchange.com/questions/100655/cosine-esque-function-with-flat-peaks-and-valleys
__device__ float getAnimationValue(float min, float max, float parameter, float time, float timescale) 
{
	float offset = (max-min)/2;
	float avg = (max+min)/2;
	return avg + offset * sinf(time*timescale)*sqrt((1+pow(parameter,2))/(1+pow(parameter*sinf(time*timescale), 2)));
}

// Steps through the the algorithm and determines if we have escaped or not.
// If we have escaped, color the square red. Otherwise color black.
// Fills the data into the single-dimensional pixel_array
__global__ void checkEscapeGPU(int width, int height, float* pixel_array, float time) 
{ 
	// Orange color: 255, 90, 0
	// Green color: 9, 255, 0
	float red = getAnimationValue(255,9, 10, time, TIME_SCALE/4);
	float green = getAnimationValue(90,255, 10, time, TIME_SCALE/4);
	float blue =0; // No blue for halloween :(
				   //
	float falloff_factor = getAnimationValue(0.5,0.9,0.7, time, TIME_SCALE);
	// Pixel Coordinates come from the initial index
	int pixel_x = blockIdx.x*blockDim.x + threadIdx.x;
	int pixel_y = blockIdx.y*blockDim.y + threadIdx.y;
		
	// (We can use the "or equal to" becuase of 0 indexing)
	// Check if IDX is out of bounds of out pixel map
	if(pixel_x >= width || pixel_y >= height) return;

	float real_component;
	float imaginary_component;
	
	//we want to color everything red
	int color_index = getPixelIndex(pixel_x, pixel_y, 0, width);
	
	real_component = getNumFromPX(pixel_x, width, X_LOWER_BOUND, X_UPPER_BOUND);
	imaginary_component = getNumFromPX(pixel_y, height, Y_LOWER_BOUND, Y_UPPER_BOUND);

	int count = 0;
	float intensity =0;
	while (count< MAX_ITERATIONS) 
	{
		intensity = pow((float)count/(float)MAX_ITERATIONS, falloff_factor); //Max value is 1
		if(getMagnitude(real_component, imaginary_component) > MAX_MAGNITUDE)
		{
		
			// If we escape, we get a color
			pixel_array[color_index]   = red/255*intensity;	
			pixel_array[color_index+1] = green/255*intensity;	
			pixel_array[color_index+2] = blue/255*intensity;	
			return;
		}

		step(&real_component, &imaginary_component, A, B);
		count++;
	}

	// If we never escape, we get colored black.
	pixel_array[color_index]   = 0;	
	pixel_array[color_index+1] = 0;	
	pixel_array[color_index+2] = 0;	
}


// Callback function for glut
void display(void) 
{ 
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels_CPU); 
	glutSwapBuffers(); 
}
void timer (int value) 
{
	long long time_elapsed = get_current_time_ms() - start_time;
	checkEscapeGPU<<<gridSize,blockSize>>>(WIDTH, HEIGHT, pixels_GPU, time_elapsed);
	CHECK();
	cudaDeviceSynchronize();

	cudaMemcpy(pixels_CPU, pixels_GPU,
			WIDTH*HEIGHT*3*sizeof(float),
			cudaMemcpyDeviceToHost);
	glutPostRedisplay();

	glutTimerFunc(16,timer,0);
}
