NVCC = nvcc
NVCCFLAGS = -std=c++11 -I../../Utility

GPU: GPU.o
	$(NVCC) $(NVCCFLAGS) -o GPU GPU.o

GPU.o:
	$(NVCC) $(NVCCFLAGS) -c -Xptxas -O3 -o GPU.o GPU.cu

clean:
	rm *.o GPU
