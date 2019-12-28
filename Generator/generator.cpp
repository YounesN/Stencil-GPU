#include <iostream>
#include <fstream>
#include <stdlib.h>

/* Header files for mkdir */
#if defined(__linux__) || defined(__APPLE__)
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#endif

using namespace std;

bool CreateDirectory(string directory);

int main(int argc, char *argv[])
{
  /* Define variables */
  int i, j, size, stride, length;
  /* initialize random seed: */
  srand (time(NULL));

  /* Check if we have the required arguments */
  if(argc < 3) {
    cerr << "Usage: ./generator <size> <stride>\n";
    cerr << "The dimension in each side = 2^size + 2 * stride\n";
    exit(EXIT_FAILURE);
  }

  /* Set the arguments into variables */
  size   = atoi(argv[1]);
  stride = atoi(argv[2]);

  /* Create Data directory in root folder */
  if(!CreateDirectory("../Data/")) {
    if(errno == EEXIST) {
      cout << "Data directory already exists. Which is fine!\n";
    } else {
      cerr << "Couldn't create directory!\n";
      exit(EXIT_FAILURE);
    }
  }

  /* Setting length of each dimension */
  length           = 1 << size;
  length          += 2 * stride;

  /* Open output file */
  ofstream fp;
  string filename  = "../Data/data_";
  filename        += to_string(size) + "_" + to_string(stride) + ".dat";
  fp.open(filename, ios::out);
  
  for(i=0; i<length; i++) {
    for(j=0; j<length; j++) {
      fp << rand() % 100 << " ";
    }
    fp << "\n";
  }

  /* Close file */
  fp.close();
  return 0;
}

/* This function will create a directory.
 * Returns "true" if successful
 * Returns "false" if unsuccessful
 * Only works on macos or Linux for now.
 */
bool CreateDirectory(string directory)
{
#if defined(__linux__) || defined(__APPLE__)
  if (mkdir(directory.c_str(), 0777) == -1) {
    return false;
  } else {
    return true;
  }
#else
  cerr << "Windows is not supported yet!\n";
  return false;
#endif
}