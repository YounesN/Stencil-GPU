NVCC = nvcc
CXXFLAGS = -Wall -O3 -I../../Utility -std=c++11
NVCCFLAGS = -std=c++11

GPU.o:
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) -c -o GPU.o GPU.cu

GPU: GPU.o
	$(NVCC) $(CXXFLAGS) $(NVCCFLAGS) -o GPU GPU.o

clean:
	rm *.o GPU