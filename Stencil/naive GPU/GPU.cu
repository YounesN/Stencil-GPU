#include <iostream>
#include <fstream>
#include <cuda.h>
#include <cuda_runtime.h>
#include "timing.h"

using namespace std;

#define from2Dto1D(arr, x, y, length) ((arr)[(y)*length+(x)])
#define BLOCKX 32
#define BLOCKY 32

/* CPU Functions */
void stencil(int **dev_input, int **dev_output, int size, int stride, int length, int time);
void run_single_stencil(int *dev_input, int *dev_output, int true_size, int stride, int length);
inline int stencil_cross(int *arr, int x, int y, int length, int order);
void read_input(int **input, int **output, string filename, int length);
void write_output(int *output, string filename, int length);

/* GPU Functions */
void copy_input_to_gpu(int *input, int **dev_input, int **dev_output, int length);

int main(int argc, char *argv[])
{
  /* Define variables */
  int *input, *output;
  int *dev_input, *dev_output;
  int size, stride, length, time;
  string filename, output_filename;
  MyTimer timer;

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
  output_filename  = "../../Data/cpu_";
  output_filename += to_string(size) + "_" + to_string(stride) + ".dat";

  /* Read data from input file */
  read_input(&input, &output, filename, length);

  /* Copy data to GPU */
  copy_input_to_gpu(input, &dev_input, &dev_output, length);

  /* Run Stencil */
  timer.StartTimer();
  stencil(&dev_input, &dev_output, size, stride, length, time);
  timer.StopTimer();

  /* Print duration */
  cout << "It took " << timer.GetDurationInSecondsAccurate() << " seconds to run!\n";

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

void stencil(int **dev_input, int **dev_output, int size, int stride, int length, int time)
{
  /* Define variables */
  int i;
  int **swap;
  int true_size = 1 << size;

  /* Loop over time dimension */
  for(i=0; i<time; i++) {
    /* Calculate block and grid sizes */
    dim3 block_size = dim3(BLOCKX, BLOCKY);
    dim3 grid_size = dim3((int)(length / BLOCKX) + 1, (int)(length / BLOCKY) + 1);
    run_single_stencil<<< grid_size, block_size >>>(*dev_input, *dev_output, true_size, stride, length);

    /* Swap pointers after each run so dev_output will always be output,
     * and dev_input will be always input
     */
    swap = dev_input;
    dev_input = dev_output;
    dev_output = swap;
  }
}

__global__ void run_single_stencil(int *dev_input, int *dev_output, int true_size, int stride, int length)
{
  /* Define variables */
  int i, j;

  /* Calculate indeces */
  threadX = blockIdx.x * blockDim.x + threadIdx.x;
  threadY = blockIdx.y * blockDim.y + threadIdx.y;

  /* Make sure indeces are not out of bound */
  if(threadX >= length || threadY >= length)
    return;

  /* Run single element stencil on all elements */
  from2Dto1D(dev_output, threadX, threadY, length) = stencil_cross(dev_input, threadX, threadX, length, stride);
}

__device__ int stencil_cross(int *arr, int x, int y, int length, int stride)
{
  /* Define variables */
  int sum = 0, i;

  /* Add cross pattern */
  for(i=-stride; i<=stride; i++) {
    sum += from2Dto1D(arr, x+i, y, length);
    sum += from2Dto1D(arr, x, y+i, length);
  }
  
  /* Counted center element twice, so substract it once */
  sum -= from2Dto1D(arr, x, y, length);

  /* Divide it by the number of elements */
  return sum / (stride * 4 + 1);
}

void read_input(int **input, int **output, string filename, int length)
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
  *input = new int[length * length];
  *output = new int[length * length];

  /* Read data from file */
  for(i=0; i<length; i++) {
    for(j=0; j<length; j++) {
      fp >> from2Dto1D(*input, i, j, length);
    }
  }
}

void write_output(int *output, string filename, int length)
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
      fp << from2Dto1D(output, i, j, length);
    }
  }
}

void copy_input_to_gpu(int *input, int **dev_input, int **dev_output, int length)
{
  /* Allocate GPU memory for input and output arrays */
  cudaMalloc((void**) dev_input, length * length * sizeof(int));
  cudaMalloc((void**) dev_output, length * length * sizeof(int));

  /* Copy input array to GPU */
  cudaMemcpy(input, *dev_input, length * length * sizeof(int), cudaMemcpyHostToDevice);
}