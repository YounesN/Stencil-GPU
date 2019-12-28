NVCC = nvcc
NVCCFLAGS = -std=c++11

GPU.o:
	$(NVCC) $(NVCCFLAGS) -c -o GPU.o GPU.cu

GPU: GPU.o
	$(NVCC) $(NVCCFLAGS) -o GPU GPU.o

clean:
	rm *.o GPU