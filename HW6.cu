// Name: Phil Alcorn
// Simple Julia CPU.
// nvcc HW6.cu -o temp -lglut -lGL
// glut and GL are openGL libraries.
/*
 What to do:
 This code displays a simple Julia fractal using the CPU.
 Rewrite the code so that it uses the GPU to create the fractal. 
 Keep the window at 1024 by 1024.
*/

// Include files
#include <stdio.h>
#include <GL/glut.h>

// Defines
#define MAX_MAGNITUDE 10.0 // If you grow larger than this, we assume that you have 
					//escaped.
#define MAX_ITERATIONS 200  // If you have not escaped after this many attempts, 
						   //we assume you are not going to escape.
#define A  -0.824	//Real part of C
#define B  -0.1711	//Imaginary part of C Global variables unsigned int 

#define WIDTH 1024
#define HEIGHT 1024
#define CHECK cudaErrorCheck(__FILE__, __LINE__)

//Global Variables (bad)


dim3 BlockSize;
dim3 GridSize;

float XMin = -2.0;
float XMax =  2.0;
float YMin = -2.0;
float YMax =  2.0;

// Function prototypes
void cudaErrorCheck(const char*, int);
void setUpDevices();
float escapeOrNotColor(float, float);
__global__ float escapeOrNotColorGPU(float, float);
void display();


int main(int argc, char** argv)
{ 
	setUpDevices();
	CHECK;

   	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_SINGLE);
   	glutInitWindowSize(WIDTH, HEIGHT);
	glutCreateWindow("Fractals--Man--Fractals");
   	glutDisplayFunc(display);
   	glutMainLoop();
}


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

void setUpDevices() 
{
	// One thread per pixel in a 1024x1024 grid, anything larger 
	// needs more blocks 
	BlockSize.x = 1024;
	BlockSize.y = 1024;
	BlockSize.z = 1;


	//The minimum number of blocks to hold the pixels in the x and y directions	
	GridSize.x = (WIDTH-1)/BlockSize.x + 1; 
	GridSize.y = (WIDTH-1)/BlockSize.y + 1; 
	GridSize.z = 1; // only two dimensions
}

float escapeOrNotColor (float x, float y) 
{
	float mag,tempX;
	int count;

	count = 0;
	mag = sqrt(x*x + y*y);;
	while (mag < MAX_MAGNITUDE && count < MAX_ITERATIONS) 
	{	
		tempX = x; //We will be changing the x but we need its old value to find y.
		x = x*x - y*y + A;
		y = (2.0 * tempX * y) + B;
		mag = sqrt(x*x + y*y);
		count++;
	}
	if(count < MAX_ITERATIONS) 
	{
		return(0.0);
	}
	else
	{
		return(1.0);
	}
}

__global__ float escapeOrNotColorGPU(float x, float y) 
{
	// A thread will operate on the pixel at it's 
	// threadIdx.x and threadIdx.y coordinates. Fairly straightfoward
	// mapping

	int x_pos = blockIdx.x*blockDim.x + threadIdx.x;
	int y_pos = blockIdx.y*blockDim.y + threadIdx.y;

	float mag, tempX;
	int count =0; // How many times we've attempted to escape
	

	//need a way to map these ID's to pixels
}

void display(void) 
{ 
	float *pixels; 
	float x, y, stepSizeX, stepSizeY;
	int k;
	
	//We need the 3 because each pixel has a red, green, and blue value.
	pixels = (float *)malloc(WIDTH*HEIGHT*3*sizeof(float));
	
	stepSizeX = (XMax - XMin)/((float)WIDTH);
	stepSizeY = (YMax - YMin)/((float)HEIGHT);
	
	k=0;
	y = YMin;
	while(y < YMax) 
	{
		x = XMin;
		while(x < XMax) 
		{
			pixels[k] = escapeOrNotColor(x,y);	//Red on or off returned from color
			pixels[k+1] = 0.0; 	//Green off
			pixels[k+2] = 0.0;	//Blue off
			k=k+3;			//Skip to next pixel (3 float jump)
			x += stepSizeX;
		}
		y += stepSizeY;
	}

	//Putting pixels on the screen.
	glDrawPixels(WIDTH, HEIGHT, GL_RGB, GL_FLOAT, pixels); 
	glFlush(); 
}

