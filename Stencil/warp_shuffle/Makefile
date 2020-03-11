NVCC = nvcc
NVCCFLAGS = -std=c++11 --use_fast_math -I../../Utility

GPU: GPU.o
	$(NVCC) $(NVCCFLAGS) -o GPU GPU.o

GPU.o:
	$(NVCC) $(NVCCFLAGS) -Xptxas -O3 -c -o GPU.o GPU.cu

clean:
	rm *.o GPU