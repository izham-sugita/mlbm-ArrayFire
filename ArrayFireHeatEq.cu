/*
program     : Calling CUDA-kernel using ArrayFire v3.4 library. ArrayFire v3.4 handles
	      the boilerplate of CUDA-API.

date        : 11th Sept 2017 (11/09/2017)

coder       : Muhammad Izham a.k.a Sugita

institution : Universiti Malaysia Perlis (UniMAP)

contact     : sugita5019@gmail.com

*/

/*
How to compile:
$ nvcc -ccbin=g++ -std=c++11 -o filename filename.cu -lcuda -lcudart -lafcuda
*/

#include<arrayfire.h>
#include<cuda.h>

#include<cstdlib>
#include<cstdio>

#include "sys/time.h"
#include "time.h"

//play with the block size to find your best performance
//hint: blockDim_x should be bigger than blockDim_y
//hint: blockDim_x*blockDim_y should not exceed 1024 (gpu dependent)

#define  blockDim_x       128
#define  blockDim_y       4
#define  pi          4.0*atan(1.0)

#define  MIN2(x, y)  ((x) < (y) ? (x) : (y))

using namespace af;

//Heat equation kernel
__global__  void  cuda_diffusion2d_0
// ====================================================================
//
// program    :  CUDA device code for 2-D diffusion equation
//               for 16 x 16 block and 16 x 16 thread per 1 block
//
// date       :  Nov 07, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   float    *f,         /* dependent variable                        */
   float    *fn,        /* dependent variable                        */
   int      nx,         /* grid number in the x-direction            */
   int      ny,         /* grid number in the x-direction            */
   float    c0,         /* coefficient no.0                          */
   float    c1,         /* coefficient no.1                          */
   float    c2          /* coefficient no.2                          */
)
// --------------------------------------------------------------------
{
   int    j,    jx,   jy;
   float  fcc,  fce,  fcw,  fcs,  fcn;

   jy = blockDim.y*blockIdx.y + threadIdx.y;
   jx = blockDim.x*blockIdx.x + threadIdx.x;
   j = nx*jy + jx;

   fcc = f[j];

   if(jx == 0) fcw = fcc;
   else        fcw = f[j - 1];

   if(jx == nx - 1) fce = fcc;
   else             fce = f[j+1];

   if(jy == 0) fcs = fcc;
   else        fcs = f[j-nx];

   if(jy == ny - 1) fcn = fcc;
   else             fcn = f[j+nx];

   fn[j] = c0*(fce + fcw)
         + c1*(fcn + fcs)
         + c2*fcc;
}


float  diffusion2d
// ====================================================================
//
// purpose    :  2-dimensional diffusion equation solved by FDM
//
// date       :  May 16, 2008
// programmer :  Takayuki Aoki
// place      :  Tokyo Institute of Technology
//
(
   int      nx,         /* x-dimensional grid size                   */
   int      ny,         /* y-dimensional grid size                   */
   float    *f,         /* dependent variable                        */
   float    *fn,        /* updated dependent variable                */
   float    kappa,      /* diffusion coefficient                     */
   float    dt,         /* time step interval                        */
   float    dx,         /* grid spacing in the x-direction           */
   float    dy          /* grid spacing in the y-direction           */
)
// --------------------------------------------------------------------
{
     float     c0 = kappa*dt/(dx*dx),   c1 = kappa*dt/(dy*dy),
               c2 = 1.0 - 2.0*(c0 + c1);

dim3  grid(nx/blockDim_x,ny/blockDim_y,1),  threads(blockDim_x,blockDim_y,1);
cuda_diffusion2d_0<<< grid, threads >>>(f,fn,nx,ny,c0,c1,c2);

return (float)(nx*ny)*7.0;

}


int main()
{

struct timeval start,finish;
double duration;
float  flops=0.0;

int imax  = 256;
int jmax  = 256;
int nodes = imax*jmax;

float dx  = 1.0/((float)imax - 1);
float dy  = 1.0/((float)jmax - 1);

//initiate from host
float h_Temp[nodes];

for(int i=0; i<imax; ++i){
 for(int j=0; j<jmax; ++j){
int id = i*jmax + j; 
h_Temp[id] = sin((float)i * dx * pi )*sin( (float)j * dy * pi);
  }
}

float kappa = 0.1;
float dt = 0.20*MIN2(dx*dx,dy*dy)/kappa;
int itermax = 20601;

array Temp0(nodes, h_Temp); //initiate on gpu
array Temp1(nodes, h_Temp); 

float *d_Temp0 = Temp0.device<float>();
float *d_Temp1 = Temp1.device<float>();

sync();

/* for output data during loop */
float *h_copy = new float[nodes];
FILE *fp0;
fp0 = fopen("peak.dat","w");


gettimeofday(&start, NULL);

// time loop
for(int iter =0; iter < itermax; ++iter){
flops += diffusion2d(imax,jmax, d_Temp0, d_Temp1, kappa, dt, dx, dy);

//(
//   int      nx,         /* x-dimensional grid size                   */
//   int      ny,         /* y-dimensional grid size                   */
//   float    *f,         /* dependent variable                        */
//   float    *fn,        /* updated dependent variable                */
//   float    kappa,      /* diffusion coefficient                     */
//   float    dt,         /* time step interval                        */
//   float    dx,         /* grid spacing in the x-direction           */
//   float    dy          /* grid spacing in the y-direction           */
//)

// Use unlock to return back to ArrayFire stream. Otherwise just use the
// d_Temp0 and d_Temp1 pointer.
// Temp0.unlock();
// Temp1.unlock();

//output to file
/* Temp1.host(h_copy);
int id = (imax/2)*jmax + (jmax/2);
fprintf(fp0,"%f\t %f\n", iter*dt, h_copy[id]);*/


if(iter > 0 && iter % 100 == 0){
printf("time(%d) = %f\n", iter, (float)iter*dt);
}

//Update pointer
d_Temp0 = d_Temp1;

}
gettimeofday(&finish, NULL);
duration = ((double)(finish.tv_sec-start.tv_sec)*1000000 + (double)(finish.tv_usec-start.tv_usec)) / 1000000;
printf("Total operations : %f\n", flops);
flops = flops/(duration*1.0e06); 

printf("Elapsed time:%lf secs\n", duration);
printf("Time per loop: %lf secs\n", duration/(double)itermax);
printf("Performance : %.2f MFlops\n", flops);

fclose(fp0);
delete [] h_copy;

/*Transfering calculation data to host */
float* h_Temp0 = new float[nodes];
Temp0.host(h_Temp0);

/*Write data to file. Tecplot ASCII format. */
FILE* fp;
fp = fopen("cuda_diff.dat","w");
fprintf(fp,"variables = \"x\", \"y\", \"Temp\" \n");
fprintf(fp,"zone t=\"test\" \n");
fprintf(fp,"i=%d,\t j=%d\n", imax, jmax);
for(int i=0; i<imax; ++i){
 for(int j=0; j<jmax; ++j){
 int   id = i*jmax + j;
 float xg = (float)i*dx;
 float yg = (float)j*dy;
 fprintf(fp,"%f %f %f\n", xg, yg, h_Temp0[id]);
 }
}
fclose(fp);
delete [] h_Temp0;



}
