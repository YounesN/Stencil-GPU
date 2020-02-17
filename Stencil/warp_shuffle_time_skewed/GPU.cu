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
#define STRIDE 2
#define N 5        // N = 2 * STRIDE + 1
#define C 6        // C = (N+P-1)

__device__ bool checkArrayAccess(int x, int y, int lengthx, int lengthy, const char *file, int line) {
  
  if(x>=lengthx || y>=lengthy) {
    printf("Illegal memory access in %s line %d! x: %d, y: %d, lengthx: %d, lengthy: %d\n", file, line, x, y, lengthx, lengthy);
    return false;
  }
  return true;
}

void printVariables(int size, int time, int length, int number_of_tiles_x,
  int number_of_tiles_y, int offset_tile_x, int dep_size_x, int dep_size_y)
{
  printf("size: %d\n", size);
  printf("time: %d\n", time);
  printf("number_of_tiles_x: %d\n", number_of_tiles_x);
  printf("number_of_tiles_y: %d\n", number_of_tiles_y);
  printf("offset_tile_x: %d\n", offset_tile_x);
  printf("length: %d\n", length);
  printf("dep_size_x: %d\n", dep_size_x);
  printf("dep_size_y: %d\n", dep_size_y);
}

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

/* CPU Functions */
void stencil(DATA_TYPE **dev_input, DATA_TYPE **dev_output, int size,
  int length, int time, float selfCoefficient, float neighborCoefficient,
  int number_of_tiles_x, int number_of_tiles_y, int offset_tile_x,
  DATA_TYPE **dev_dep_up, DATA_TYPE **dev_dep_down, DATA_TYPE **dev_dep_flag,
  int dep_size_x, int dep_size_y);

__global__ void run_stencil(DATA_TYPE *dev_input, DATA_TYPE *dev_output,
  DATA_TYPE *dev_dep_up, DATA_TYPE *dev_dep_down, DATA_TYPE *dev_dep_flag,
  int dep_size_x, int dep_size_y, int offset_tile_x, int length,
  float selfCoefficient, float neighborCoefficient, int time);

void read_input(DATA_TYPE **input, DATA_TYPE **output, string filename,
  int length);

void write_output(DATA_TYPE *output, string filename, int length);

void allocateDependencyArrays(DATA_TYPE **dev_dep_up, DATA_TYPE **dev_dep_down,
  DATA_TYPE **dev_dep_flag, int dep_size_x, int dep_size_y,
  int number_of_tiles_x, int number_of_tiles_y);

/* GPU Functions */
void copy_input_to_gpu(DATA_TYPE *input, DATA_TYPE **dev_input, DATA_TYPE **dev_output, int length);

int main(int argc, char *argv[])
{
  /* Define variables */
  DATA_TYPE *input, *output;
  DATA_TYPE *dev_input, *dev_output, *dev_dep_up, *dev_dep_down, *dev_dep_flag;
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
  length           = 1 << size;       // length = 2 ^ size
  length          += 2 * STRIDE;      //        + 2 * stride
  filename         = "../../Data/data_";
  filename        += toString(size) + "_" + toString(STRIDE) + ".dat";
  output_filename  = "../../Data/gpu_shfl_skewed_";
  output_filename += toString(size) + "_" + toString(STRIDE) + ".dat";

  /* Read data from input file */
  read_input(&input, &output, filename, length);

  /* Copy data to GPU */
  copy_input_to_gpu(input, &dev_input, &dev_output, length);

  /////////// INVESTIGATE THIS PART FOR INVALID MEMORY ACCESS
  int number_of_tiles_x = int(length / (WARP_SIZE * NUMBER_OF_WARPS_PER_X - 2 * STRIDE)) + 1;
  int number_of_tiles_y = int((length-2*STRIDE) / P);
  int offset_tile_x     = WARP_SIZE - 2 * STRIDE;

  // Calculate dependency array sizes
  int dep_size_x = length;
  int dep_size_y = number_of_tiles_y * STRIDE;
  allocateDependencyArrays(&dev_dep_up, &dev_dep_down, &dev_dep_flag, dep_size_x, dep_size_y, number_of_tiles_x, number_of_tiles_y);

  printVariables(size, time, length, number_of_tiles_x, number_of_tiles_y, offset_tile_x, dep_size_x, dep_size_y);

  /* Run Stencil */
  timer.StartTimer();
  stencil(&dev_input, &dev_output, size, length, time, selfCoefficient,
    neighborCoefficient, number_of_tiles_x, number_of_tiles_y, offset_tile_x,
    &dev_dep_up, &dev_dep_down, dep_size_x, dep_size_y);
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

void stencil(DATA_TYPE **dev_input, DATA_TYPE **dev_output, int size,
  int length, int time, float selfCoefficient, float neighborCoefficient,
  int number_of_tiles_x, int number_of_tiles_y, int offset_tile_x,
  DATA_TYPE **dev_dep_up, DATA_TYPE **dev_dep_down, DATA_TYPE **dev_dep_flag,
  int dep_size_x, int dep_size_y)
{
  /* Calculate block and grid sizes */
  dim3 block_size = dim3(WARP_SIZE * NUMBER_OF_WARPS_PER_X, 1, 1);
  dim3 grid_size = dim3(number_of_tiles_x, number_of_tiles_y, 1);
  run_stencil<<< grid_size, block_size >>>(*dev_input, *dev_output, *dev_dep_up,
    *dev_dep_down, *dev_dep_flag, dep_size_x, dep_size_y,
    offset_tile_x, length, selfCoefficient, neighborCoefficient, time);
  // gpuErrchk(cudaGetLastError());
}

__global__ void run_stencil(DATA_TYPE *dev_input, DATA_TYPE *dev_output,
  DATA_TYPE *dev_dep_up, DATA_TYPE *dev_dep_down, DATA_TYPE *dev_dep_flag,
  int dep_size_x, int dep_size_y, int offset_tile_x, int length,
  float selfCoefficient, float neighborCoefficient, int time)
{
  /* Declare variables */
  int i, j, t;
  DATA_TYPE v[C], o[C];
  int offset_x = blockIdx.x * (offset_tile_x * NUMBER_OF_WARPS_PER_X) +
                 (threadIdx.x / 32) * offset_tile_x;
  int offset_y = blockIdx.y * P;
  int lane     = threadIdx.x % WARP_SIZE;
  int tileX = blockIdx.x;
  int tileY = blockIdx.y;

  int lanePlusOffsetX = lane + offset_x;
  if(lanePlusOffsetX >= length) {
    return;
  }

  /* Initialize v[] array */
  for(i=0; i<C; i++) {
    v[i] = dev_input[from2Dto1D(lanePlusOffsetX, i + offset_y, length)];
  }

  for(t=1; t<=time; t++) {
    /* Main loop calculates for all P elements */
    if(i%2==0) { // switch between shuffle up and down to keep data close to origin
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
        o[i] = sum;
      }
    } else {
      for(i=STRIDE; i<P+STRIDE; i++) {
        DATA_TYPE sum = 0;
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
    }
    for(i=0; i<C; i++) {
      v[i] = o[i];
    }

    for(i=0; i<STRIDE; i++) {
      dev_dep_down[from2Dto1D(lanePlusOffsetX, offset_y+i, length)] = v[P+STRIDE+i];
    }

    for(i=0; i<STRIDE; i++) {
      dev_dep_up[from2Dto1D(lanePlusOffsetX, offset_y+i, length)] = v[i];
    }

    dev_dep_flag[from2Dto1D(tileX, tileY, number_of_tiles_x)] = t;
    while(dev_dep_flag[from2Dto1D(tileX, tileY-1, number_of_tiles_x)] != t-1);
    while(dev_dep_flag[from2Dto1D(tileX, tileY+1, number_of_tiles_x)] != t-1);

    for(i=0; i<STRIDE; i++) {
      v[P+STRIDE+i] = dev_dep_up[from2Dto1D(lanePlusOffsetX, offset_y+i, length)];
    }

    for(i=0; i<STRIDE; i++) {
      v[i] = dev_dep_down[from2Dto1D(lanePlusOffsetX, offset_y+i, length)] = v[i];
    }
  }

  /* Write the sum back to global memory */
  for(i=STRIDE; i<P+STRIDE; i++) {
    if(lane >= 2*STRIDE && lane+offset_x < length && i+offset_y < length-STRIDE) {
      dev_output[from2Dto1D(lane+offset_x-STRIDE, i+offset_y, length)] = o[i];
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

void allocateDependencyArrays(DATA_TYPE **dev_dep_up, DATA_TYPE **dev_dep_down,
  DATA_TYPE **dev_dep_flag, int dep_size_x, int dep_size_y,
  int number_of_tiles_x, int number_of_tiles_y);
{
  gpuErrchk(cudaMalloc((void**) dev_dep_up, dep_size_x * dep_size_y * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMalloc((void**) dev_dep_down, dep_size_x * dep_size_y * sizeof(DATA_TYPE)));

  gpuErrchk(cudaMalloc((void**) dev_dep_flag, number_of_tiles_x * number_of_tiles_y * sizeof(DATA_TYPE)));
  gpuErrchk(cudaMemset(*dev_dep_flag, 0, number_of_tiles_x * number_of_tiles_y * sizeof(DATA_TYPE)));
}