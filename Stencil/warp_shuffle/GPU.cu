#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timing.h"

using namespace std;

#define from2Dto1D(x, y, length) ((y)*length+(x))
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define DATA_TYPE float

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* CPU Functions */
void stencil(DATA_TYPE **dev_input, DATA_TYPE **dev_output, int size, int stride, int length, int time, float selfCoefficient, float neighborCoefficient);
__global__ void run_single_stencil(DATA_TYPE *dev_input, DATA_TYPE *dev_output, const int C, int offset_tile_x, int offset_tile_y, int length, int stride, int P, float selfCoefficient, float neighborCoefficient, int number_of_warps_x);
void read_input(DATA_TYPE **input, DATA_TYPE **output, string filename, int length);
void write_output(DATA_TYPE *output, string filename, int length);

/* GPU Functions */
void copy_input_to_gpu(DATA_TYPE *input, DATA_TYPE **dev_input, DATA_TYPE **dev_output, int length);

int main(int argc, char *argv[])
{
  /* Define variables */
  DATA_TYPE *input, *output;
  DATA_TYPE *dev_input, *dev_output;
  int size, stride, length, time;
  string filename, output_filename;
  MyTimer timer;
  float selfCoefficient = 1.0/9.0;
  float neighborCoefficient = 1.0/9.0;

  /* Check if the arguments are set */
  if(argc < 4) {
    cerr << "Usage: ./CPU <size> <stride> <time>\n";
    exit(EXIT_FAILURE);
  }

  /* Set initial variables */
  size             = atoi(argv[1]);
  stride           = atoi(argv[2]);
  time             = atoi(argv[3]);
  length           = 1 << size;       // length = 2 ^ size
  length          += 2 * stride;      //        + 2 * stride
  filename         = "../../Data/data_";
  filename        += to_string(size) + "_" + to_string(stride) + ".dat";
  output_filename  = "../../Data/gpu_shfl_";
  output_filename += to_string(size) + "_" + to_string(stride) + ".dat";

  /* Read data from input file */
  read_input(&input, &output, filename, length);

  /* Copy data to GPU */
  copy_input_to_gpu(input, &dev_input, &dev_output, length);

  /* Run Stencil */
  timer.StartTimer();
  stencil(&dev_input, &dev_output, size, stride, length, time, selfCoefficient, neighborCoefficient);
  cudaDeviceSynchronize();
  timer.StopTimer();

  /* Print duration */
  cout << "It took " << timer.GetDurationInSecondsAccurate() << " seconds to run!\n";

  /* Copy data back to CPU */
  gpuErrchk(cudaMemcpy(output, dev_output, length * length * sizeof(int), cudaMemcpyDeviceToHost));

  /* Output data */
  write_output(output, output_filename, length);

  /* Free allocated memory */
  cudaFree(dev_input);
  cudaFree(dev_output);
  delete [] input;
  delete [] output;

  /* End of program */
  return 0;
}

void stencil(DATA_TYPE **dev_input, DATA_TYPE **dev_output, int size, int stride, int length, int time, float selfCoefficient, float neighborCoefficient)
{
  /* Define variables */
  int i;
  DATA_TYPE **swap;

  /* System variables */
  int P                 = 2;                // P: defines the number of cell each thread calculates
  int N                 = (2 * stride) + 1;  // N: stencil length each direction
  int C                 = (N + P - 1);       // C: each block will calculate (warp_size * C) size
  int number_of_warps_x = 32

  int number_of_tiles_x = int(length / (WARP_SIZE * number_of_warps_x - 2 * stride)) + 1;
  int number_of_tiles_y = int(length / P) + 1;
  int offset_tile_x     = WARP_SIZE - 2 * stride;
  int offset_tile_y     = P;

  /* Loop over time dimension */
  for(i=0; i<time; i++) {
    /* Calculate block and grid sizes */
    dim3 block_size = dim3(WARP_SIZE * number_of_warps_x, 1, 1);
    dim3 grid_size = dim3(number_of_tiles_x, number_of_tiles_y, 1);
    run_single_stencil<<< grid_size, block_size >>>(*dev_input, *dev_output, C, offset_tile_x, offset_tile_y, length, stride, P, selfCoefficient, neighborCoefficient, number_of_warps_x);
    gpuErrchk(cudaGetLastError());
    //cudaDeviceSynchronize();

    /* Swap pointers after each run so dev_output will always be output,
     * and dev_input will be always input
     */
    swap = dev_input;
    dev_input = dev_output;
    dev_output = swap;
  }
}

__global__ void run_single_stencil(DATA_TYPE *dev_input, DATA_TYPE *dev_output, const int C, int offset_tile_x, int offset_tile_y, int length, int stride, int P, float selfCoefficient, float neighborCoefficient, int number_of_warps_x)
{
  /* Declare variables */
  int i, j;
  DATA_TYPE v[6], o[6];
  int offset_x = blockIdx.x * (offset_tile_x * number_of_warps_x) + (threadIdx.x / 32) * offset_tile_x;
  int offset_y = blockIdx.y * offset_tile_y;
  int lane     = threadIdx.x % WARP_SIZE;

  /* Initialize v[] array */
  for(i=0; i<C; i++) {
    v[i] = dev_input[from2Dto1D(lane + offset_x, i + offset_y, length)];
  }

  /* Main loop calculates for all P elements */
  for(i=stride; i<P+stride; i++) {
    int sum = 0;

    /* Left wing */
    for(j=-stride; j<0; j++) {
      //sum = v[i] * neighborCoefficient + sum;
      sum = v[i] + sum;

      /* Shuffle up */
      sum = __shfl_up_sync(FULL_MASK, sum, 1);
    }

    /* Center column */
    for(j=-stride; j<=stride; j++) {
      sum = v[i+j] + sum;
    }

    /* Right wing */
    for(j=1; j<=stride; j++) {
      /* Shuffle up */
      sum = __shfl_up_sync(FULL_MASK, sum, 1);
      
      sum = v[i] + sum;
    }
    
    sum /= (stride * 4 + 1);
    o[i] = sum;
  }

  /* Write the sum back to global memory */
  for(i=stride; i<P+stride; i++) {
    if(lane >= 2*stride && lane+offset_x < length && i+offset_y < length-stride) {
      dev_output[from2Dto1D(lane+offset_x-stride, i+offset_y, length)] = o[i];
    }
  }
}

void read_input(DATA_TYPE **input, DATA_TYPE **output, string filename, int length)
{
  /* Define variables */
  int i, j;
  ifstream fp;

  /* Open input file */
  fp.open(filename.c_str(), ios::in);
  if(!fp) {
    cerr << "Couldn't open input file to read data!\n";
    exit(EXIT_FAILURE);
  }

  /* Allocate space for our arrays */
  *input = new DATA_TYPE[length * length];
  *output = new DATA_TYPE[length * length];

  /* Read data from file */
  for(i=0; i<length; i++) {
    for(j=0; j<length; j++) {
      fp >> (*input)[from2Dto1D(i, j, length)];
      (*output)[from2Dto1D(j, i, length)] = 0;
    }
  }
}

void write_output(DATA_TYPE *output, string filename, int length)
{
  /* Define variables */
  int i, j;
  ofstream fp;

  /* Open output file */
  fp.open(filename, ios::out);
  if(!fp) {
    cerr << "Couldn't open output file to write data!\n";
    exit(EXIT_FAILURE);
  }

  for(i=0; i<length; i++) {
    for(j=0; j<length; j++) {
      fp << output[from2Dto1D(j, i, length)] << " ";
    }
    fp << "\n";
  }
}

void copy_input_to_gpu(DATA_TYPE *input, DATA_TYPE **dev_input, DATA_TYPE **dev_output, int length)
{
  /* Allocate GPU memory for input and output arrays */
  gpuErrchk(cudaMalloc((void**) dev_input, length * length * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void**) dev_output, length * length * sizeof(DATA_TYPE)));

  /* Copy input array to GPU */
  gpuErrchk(cudaMemcpy(*dev_input, input, length * length * sizeof(DATA_TYPE), cudaMemcpyHostToDevice));
}