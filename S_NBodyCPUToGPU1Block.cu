// Name: 
// Creating a GPU nBody simulation from an nBody CPU simulation. 
// nvcc 18NBodyGPU.cu -o temp -lglut -lm -lGLU -lGL

/*
 What to do:
 This is some lean nBody code that runs on the CPU. Rewrite it, 
 keeping the same general format, 
 but offload the compute-intensive parts of the code to the GPU for 
 acceleration.
 Note: The code takes two arguments as inputs:
 1. The number of bodies to simulate, (We will keep the number of 
 bodies under 1024 for this HW so it can be run on one block.)
 2. Whether to draw sub-arrangements of the bodies during the simulation 
 (1), or only the first and last arrangements (0).
*/

/*
 Purpose:
 To learn how to move an Nbody CPU simulation to an Nbody GPU simulation..
*/

// Include files
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Defines
#define PI 3.14159265359
#define DRAW_RATE 10

// This is to create a Lennard-Jones type function G/(r^p) - H(r^q). 
// (p < q) p has to be less than q.
//
// In this code we will keep it a p = 2 and q = 4 problem. 
// The diameter of a body is found using the general
// case so it will be more robust but in the code leaving it as a set 2, 
// 4 problem make the coding much easier.
#define G 10.0
#define H 10.0
#define LJP  2.0
#define LJQ  4.0

#define DT 0.0001
#define RUN_TIME 1.0

// Globals
int N, DrawFlag;
float3 *P, *V, *F;
float3 *PGPU, *VGPU, *FGPU;
float *M; 
float *MGPU; 
float GlobeRadius, Diameter, Radius;
float Damp;

// Function prototypes
void keyPressed(unsigned char, int, int);
long elaspedTime(struct timeval, struct timeval);
void drawPicture();
void timer();
void setup();
void nBody();
void cudaErrorCheck(const char *file, int line);

__global__ void zero_forces(float3* FGPU, int N);
__global__ void calculate_forces(const float3* pos, const float* mass, 
								 float3* forces,  int N);
__global__ void step(float3 *P, float3 *V, const float3 *F, const float *M, 
                          float dt, float damp, int N, int firstStep);


int main(int argc, char** argv)
{
	if( argc < 3)
	{
		printf("\n You need to enter the number of bodies (an int)"); 
		printf("\n and if you want to draw the bodies as they move \
				(1 draw, 0 don't draw),");
		printf("\n on the comand line.\n"); 
		exit(0);
	}
	else
	{
		N = atoi(argv[1]);
		DrawFlag = atoi(argv[2]);
	}
	 if (N <= 0 || N > 1024) {
        printf("N must be in [1, 1024]\n");
        exit(1);
    }
	
	setup();
	
	int XWindowSize = 1000;
	int YWindowSize = 1000;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("nBody Test");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutKeyboardFunc(keyPressed);
	glutDisplayFunc(drawPicture);
	
	float3 eye = {0.0f, 0.0f, 2.0f*GlobeRadius};
	float near = 0.2;
	float far = 5.0*GlobeRadius;
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, near, far);
	glMatrixMode(GL_MODELVIEW);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	gluLookAt(eye.x, eye.y, eye.z, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	
	glutMainLoop();
	return 0;
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


// This functions resets all the force vectors to 0
__global__ void zero_forces(float3* F, int N)
{
	int i = threadIdx.x;
    if (i < N) F[i] = make_float3(0.f, 0.f, 0.f);
}

__global__ void calculate_forces(const float3* pos, 
								 const float* mass, 
								 float3* forces, 
								int N) 
{
	int i = threadIdx.x; // We can just use the threadid since we only have one block
    if (i >= N) return; // Only modify the first N bodies

	// Values for the ith body
	float3 pi = pos[i];
    float  mi = mass[i];
	// Start our sum at zero
    float3 fi = make_float3(0.f, 0.f, 0.f);
	
	// loop through the rest of the bodies
    for (int j = 0; j < N; ++j) {
        if (j == i) continue; // don't calculate force on the same bodies
        float3 pj = pos[j];
        float mj = mass[j];

        float dx = pj.x - pi.x;
        float dy = pj.y - pi.y;
        float dz = pj.z - pi.z;
        float r_squared = dx*dx + dy*dy + dz*dz;

        // inv distances
		// rsqrtf = 1/sqrtf 
        float invR  = rsqrtf(r_squared); // 1/r
        float invR2 = invR * invR;       // 1/r^2
        float invR4 = invR2 * invR2;     // 1/r^4

		// G/d^2 - H/d^4
        float force_mag = (G * mi * mj) * invR2 - (H * mi * mj) * invR4; 

		// multiply by 1/d for direction
        float scale = force_mag * invR;  

        fi.x += scale * dx;
        fi.y += scale * dy;
        fi.z += scale * dz;
    }
    forces[i] = fi;
}
__global__ void step(float3 *P, float3 *V, const float3 *F, const float *M, 
                          float dt, float damp, int N, int firstStep)
{
    int i = threadIdx.x;
    if (i >= N) return;

	//calculate 1/mass so we don't have to keep dividing
    float invMi = 1.0f / M[i];
    float3 Vi = V[i]; // get ith velocity
    float3 Fi = F[i]; // get ith force

	float half;
	if (firstStep) { half = 0.5; } 
	if (!firstStep) { half = 1; }
	Vi.x += ((Fi.x - damp * Vi.x) * invMi) * dt * half;
	Vi.y += ((Fi.y - damp * Vi.y) * invMi) * dt * half;
	Vi.z += ((Fi.z - damp * Vi.z) * invMi) * dt * half;
	
	// now update the positions with the new velocity
    float3 Pi = P[i];
    Pi.x += Vi.x * dt;
    Pi.y += Vi.y * dt;
    Pi.z += Vi.z * dt;

    V[i] = Vi; // replace the old values with the new values
    P[i] = Pi;
}

void keyPressed(unsigned char key, int x, int y)
{
	if(key == 's')
	{
		timer();
	}
	
	if(key == 'q')
	{
		exit(0);
	}
}

// Calculating elasped time.
long elaspedTime(struct timeval start, struct timeval end)
{
	// tv_sec = number of seconds past the Unix epoch 01/01/1970
	// tv_usec = number of microseconds past the current second.
	
	long startTime = start.tv_sec * 1000000 + start.tv_usec; // In microseconds.
	long endTime = end.tv_sec * 1000000 + end.tv_usec; // In microseconds

	// Returning the total time elasped in microseconds
	return endTime - startTime;
}

void drawPicture()
{
	int i;
	
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	glColor3d(1.0,1.0,0.5);
	for(i=0; i<N; i++)
	{
		glPushMatrix();
		glTranslatef(P[i].x, P[i].y, P[i].z);
		glutSolidSphere(Radius,20,20);
		glPopMatrix();
	}
	
	glutSwapBuffers();
}

void timer()
{	
	timeval start, end;
	long computeTime;
	
	drawPicture();
	gettimeofday(&start, NULL);
    nBody();
    gettimeofday(&end, NULL);
    drawPicture();
    	
	computeTime = elaspedTime(start, end);
	printf("\n The compute time was %ld microseconds.\n\n", computeTime);
}

void setup()
{
	float randomAngle1, randomAngle2, randomRadius;
	float d, dx, dy, dz;
	int test;
	
	Damp = 0.5f;
	
	M = (float*)malloc(N*sizeof(float));
    cudaMallocHost(&P, N*sizeof(float3));	
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocHost(&V, N*sizeof(float3));	
	cudaErrorCheck(__FILE__, __LINE__);
    cudaMallocHost(&F, N*sizeof(float3));	
	cudaErrorCheck(__FILE__, __LINE__);
	
	Diameter = pow(H/G, 1.0/(LJQ - LJP)); // This is the value where the 
										  // force is zero for the L-J type force.
	Radius = Diameter/2.0;
	
	// Using the radius of a body and a 68% packing ratio to find the 
	// radius of a global sphere that should hold all the bodies.
	// Then we double this radius just so we can get all the bodies setup 
	// with no problems. 
	float totalVolume = float(N)*(4.0/3.0)*PI*Radius*Radius*Radius;
	totalVolume /= 0.68;
	float totalRadius = pow(3.0*totalVolume/(4.0*PI), 1.0/3.0);
	GlobeRadius = 2.0*totalRadius;
	
	// Randomly setting these bodies in the glaobal sphere and setting the 
	// initial velosity, inotial force, and mass.
	for(int i = 0; i < N; i++)
	{
		test = 0;
		while(test == 0)
		{
			// Get random position.
			randomAngle1 = ((float)rand()/(float)RAND_MAX)*2.0*PI;
			randomAngle2 = ((float)rand()/(float)RAND_MAX)*PI;
			randomRadius = ((float)rand()/(float)RAND_MAX)*GlobeRadius;
			P[i].x = randomRadius*cos(randomAngle1)*sin(randomAngle2);
			P[i].y = randomRadius*sin(randomAngle1)*sin(randomAngle2);
			P[i].z = randomRadius*cos(randomAngle2);
			
			// Making sure the balls centers are at least a diameter apart.
			// If they are not throw these positions away and try again.
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = P[i].x-P[j].x;
				dy = P[i].y-P[j].y;
				dz = P[i].z-P[j].z;
				d = sqrt(dx*dx + dy*dy + dz*dz);
				if(d < Diameter)
				{
					test = 0;
					break;
				}
			}
		}
	
		V[i].x = 0.0;
		V[i].y = 0.0;
		V[i].z = 0.0;
		
		F[i].x = 0.0;
		F[i].y = 0.0;
		F[i].z = 0.0;
		
		M[i] = 1.0;

		}
	// Malloc Device arrays, copy data over
	cudaMalloc(&PGPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&VGPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&FGPU, N*sizeof(float3));
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMalloc(&MGPU, N*sizeof(float));
	cudaErrorCheck(__FILE__, __LINE__);

	cudaMemcpy(PGPU, P, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(VGPU, V, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(FGPU, F, N * sizeof(float3), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);
	cudaMemcpy(MGPU, M, N * sizeof(float), cudaMemcpyHostToDevice);
	cudaErrorCheck(__FILE__, __LINE__);

	printf("\n To start timing type s.\n");
}

void nBody()
{
	int    drawCount = 0; 
	float  time = 0.0;
	float dt = DT;
	
	dim3 gridSize, blockSize;
	gridSize.x = 1;
	gridSize.y = 1;
	gridSize.z = 1;

	blockSize.x = 1024;
	blockSize.y = 1;
	blockSize.z = 1;

	int firstStep=1;

	while(time < RUN_TIME)
	{
		zero_forces<<<gridSize, blockSize>>>(FGPU, N);
		cudaErrorCheck(__FILE__,__LINE__);
		cudaDeviceSynchronize();

		calculate_forces<<<gridSize, blockSize>>>(PGPU, MGPU, FGPU, N);
		cudaErrorCheck(__FILE__,__LINE__);
		cudaDeviceSynchronize();


		step<<<gridSize, blockSize>>>(PGPU, VGPU, FGPU, MGPU, dt, Damp, N, firstStep);
		cudaErrorCheck(__FILE__,__LINE__);
		cudaDeviceSynchronize();


		firstStep=0;

		if(drawCount == DRAW_RATE) 
		{
			if(DrawFlag) 
			{
				cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
				cudaErrorCheck(__FILE__, __LINE__);
				drawPicture();
			}
			drawCount = 0;
		}
		
		time += dt;
		drawCount++;
	}
	cudaMemcpy(P, PGPU, N*sizeof(float3), cudaMemcpyDeviceToHost);
	cudaErrorCheck(__FILE__, __LINE__);
}


