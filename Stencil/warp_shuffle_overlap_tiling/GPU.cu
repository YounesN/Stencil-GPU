#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timing.h"
#include <algorithm>

using namespace std;

#define from2Dto1D(x, y, length) ((y)*length+(x))
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define DATA_TYPE float
#define NUMBER_OF_WARPS_PER_X 1
#define P 2
#define STRIDE 3
#define N 7        // N = 2 * STRIDE + 1
#define C 8        // C = (N+P-1)
#define BLOCKT 2   // How many tiles in T dimensions

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

string toString(int n)
{
    string tmp = "";
    while (n > 0) {
        int re = n % 10;
        n = n / 10;
        tmp += re + '0';
    }
    reverse(tmp.begin(), tmp.end());
    return tmp;
}

__global__ void run_single_stencil(DATA_TYPE *dev_input, DATA_TYPE *dev_output,
  int length, DATA_TYPE selfCoefficient, DATA_TYPE neighborCoefficient)
{
  /* Declare variables */
  int i, j;
  DATA_TYPE v[C], o[C];
  // each block thread ID will start with a multiplication of this number:
  int compute_x_size  = WARP_SIZE - 2 * STRIDE * BLOCKT;
  int offset_x        = blockIdx.x * compute_x_size;
  int offset_y        = blockIdx.y * P;
  int lane            = threadIdx.x % WARP_SIZE;
  int lanePlusOffsetX = offset_x + threadIdx.x;

  /* Initialize v[] array */
  for(i=0; i<C; i++) {
    v[i] = dev_input[from2Dto1D(lanePlusOffsetX, i + offset_y, length)];
  }

  /* Main loop calculates for all P elements */
  #pragma unroll
  for(i=STRIDE; i<P+STRIDE; i++) {
    DATA_TYPE sum = 0;

    /* Left wing */
    for(j=-STRIDE; j<0; j++) {
      sum = v[i] * neighborCoefficient + sum;

      /* Shuffle up */
      sum = __shfl_up_sync(FULL_MASK, sum, 1);
    }

    /* Center column */
    for(j=-STRIDE; j<=STRIDE; j++) {
      if(j == 0)
        sum = v[i+j] * selfCoefficient + sum;
      else
        sum = v[i+j] * neighborCoefficient + sum;
    }

    /* Right wing */
    for(j=1; j<=STRIDE; j++) {
      /* Shuffle up */
      sum = __shfl_up_sync(FULL_MASK, sum, 1);
      sum = v[i] * neighborCoefficient + sum;
    }
    
    v[i] = sum;

    sum = 0;

    /* Left wing */
    for(j=-STRIDE; j<0; j++) {
      sum = v[i] * neighborCoefficient + sum;

      /* Shuffle up */
      sum = __shfl_down_sync(FULL_MASK, sum, 1);
    }

    /* Center column */
    for(j=-STRIDE; j<=STRIDE; j++) {
      if(j == 0)
        sum = v[i+j] * selfCoefficient + sum;
      else
        sum = v[i+j] * neighborCoefficient + sum;
    }

    /* Right wing */
    for(j=1; j<=STRIDE; j++) {
      /* Shuffle up */
      sum = __shfl_down_sync(FULL_MASK, sum, 1);
      sum = v[i] * neighborCoefficient + sum;
    }
    
    o[i] = sum;
  }

  /* Write the sum back to global memory */
  for(i=STRIDE; i<P+STRIDE; i++) {
    if(lane >= 2*STRIDE && lanePlusOffsetX < length && i+offset_y < length-STRIDE) {
      dev_output[from2Dto1D(lanePlusOffsetX+BLOCKT*STRIDE, i+offset_y, length)] = o[i];
    }
  }
}

void stencil(DATA_TYPE **dev_input, DATA_TYPE **dev_output, int size,
  int length, int time, float selfCoefficient, float neighborCoefficient)
{
  /* Define variables */
  int i;
  DATA_TYPE **swap;

  // total number of available threads     = warpsize * number_of_warps_per_x 
  // extra data we need to load per t      = 2 * stride (stride per direction, left and right)
  // number of exact cell each block will process:
  int number_of_tiles_x = int(length / (WARP_SIZE * NUMBER_OF_WARPS_PER_X - 2 * STRIDE * BLOCKT)) + 1;

  // each thread will process P cell in y direction
  // so the number of tiles in y direction will be:
  int number_of_tiles_y = int(length / P) + 1;
  
  /* Calculate block and grid sizes */
  dim3 block_size = dim3(WARP_SIZE * NUMBER_OF_WARPS_PER_X, 1, 1);
  dim3 grid_size = dim3(number_of_tiles_x, number_of_tiles_y, 1);

  /* Loop over time dimension */
  for(i=0; i<time / BLOCKT; i++) {
    run_single_stencil<<< grid_size, block_size >>>(*dev_input, *dev_output,
      length, selfCoefficient, neighborCoefficient);
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

int main(int argc, char *argv[])
{
  /* Define variables */
  DATA_TYPE *input, *output;
  DATA_TYPE *dev_input, *dev_output;
  int size, length, time;
  string filename, output_filename;
  MyTimer timer;
  float selfCoefficient = 1.0/9.0;
  float neighborCoefficient = 1.0/9.0;

  /* Check if the arguments are set */
  if(argc < 3) {
    cerr << "Usage: ./CPU <size> <time>\n";
    exit(EXIT_FAILURE);
  }

  /* Set initial variables */
  size             = atoi(argv[1]);
  time             = atoi(argv[2]);
  if(time % BLOCKT != 0) {
    cerr << "Please time that is divisible by " << BLOCKT << "!\n";
    exit(EXIT_FAILURE);
  }
  length           = 1 << size;       // length = 2 ^ size
  length          += 2 * STRIDE;      //        + 2 * stride
  filename         = "../../Data/data_";
  filename        += toString(size) + "_" + toString(STRIDE) + ".dat";
  output_filename  = "../../Data/gpu_shfl_overlap_";
  output_filename += toString(size) + "_" + toString(STRIDE) + ".dat";

  /* Read data from input file */
  read_input(&input, &output, filename, length);

  /* Copy data to GPU */
  copy_input_to_gpu(input, &dev_input, &dev_output, length);

  /* Run Stencil */
  timer.StartTimer();
  stencil(&dev_input, &dev_output, size, length, time, selfCoefficient, neighborCoefficient);
  cudaDeviceSynchronize();
  timer.StopTimer();

  /* Print duration */
  cout << "It took " << timer.GetDurationInSecondsAccurate() << " seconds to run!\n";

  /* Copy data back to CPU */
  gpuErrchk(cudaMemcpy(output, dev_output, length * length * sizeof(DATA_TYPE), cudaMemcpyDeviceToHost));

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