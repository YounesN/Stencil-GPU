NVCC = nvcc
NVCCFLAGS = -std=c++11 -I../../Utility -g -G

GPU: GPU.o
	$(NVCC) $(NVCCFLAGS) -o GPU GPU.o

GPU.o:
	$(NVCC) $(NVCCFLAGS) -Xptxas -c -o GPU.o GPU.cu

clean:
	rm *.o GPU
