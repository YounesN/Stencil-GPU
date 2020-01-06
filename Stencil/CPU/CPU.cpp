#include <iostream>
#include <fstream>
#include "timing.h"

using namespace std;

#define from2Dto1D(arr, x, y, length) ((arr)[(y)*length+(x)])

void stencil(int **input, int **output, int size, int stride, int length, int time);
void run_single_stencil(int *input, int *output, int true_size, int stride, int length);
inline int stencil_cross(int *arr, int x, int y, int length, int order);
void read_input(int **input, int **output, string filename, int length);
void write_output(int *output, string filename, int length);

int main(int argc, char *argv[])
{
  /* Define variables */
  int *input, *output;
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

  /* Run Stencil */
  timer.StartTimer();
  stencil(&input, &output, size, stride, length, time);
  timer.StopTimer();

  /* Print duration */
  cout << "It took " << timer.GetDurationInSecondsAccurate() << " seconds to run!\n";

  /* Output data */
  write_output(output, output_filename, length);

  /* Free allocated memory */
  delete [] input;
  delete [] output;

  /* End of program */
  return 0;
}

void stencil(int **input, int **output, int size, int stride, int length, int time)
{
  /* Define variables */
  int i;
  int **swap;
  int true_size = 1 << size;

  /* Loop over time dimension */
  for(i=0; i<time; i++) {
    run_single_stencil(*input, *output, true_size, stride, length);

    /* Swap pointers after each run so output will always be output,
     * and input will be always input
     */
    swap = input;
    input = output;
    output = swap;
  }
}

void run_single_stencil(int *input, int *output, int true_size, int stride, int length)
{
  /* Define variables */
  int i, j;

  /* Run single element stencil on all elements */
  for(i=stride; i<true_size+stride; i++) {
    for(j=stride; j<true_size+stride; j++) {
      from2Dto1D(output, i, j, length) = stencil_cross(input, i, j, length, stride);
    }
  }
}

inline int stencil_cross(int *arr, int x, int y, int length, int stride)
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
      fp << from2Dto1D(output, j, i, length) << " ";
    }
    fp << "\n";
  }
}