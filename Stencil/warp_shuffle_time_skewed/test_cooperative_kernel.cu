#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;

int main() {
  int pi=0;
  CUdevice dev;
  cuDeviceGet(&dev,0); // get handle to device 0
  cuDeviceGetAttribute(&pi, CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH, dev);

  cout << "pi: " << pi << endl;

  return 0;
}
