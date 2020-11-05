NVCC = nvcc
NVCCFLAGS = -std=c++11 -I../../Utility

GPU: GPU.o
	$(NVCC) $(NVCCFLAGS) -o GPU GPU.o

GPU.o:
	$(NVCC) $(NVCCFLAGS) -c -o GPU.o GPU.cu -g -G

clean:
	rm *.o GPU
