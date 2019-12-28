CXX = g++
CXXFLAGS = -Wall -O3 -I../../Utility -std=c++11

generator: CPU.o
	$(CXX) $(CXXFLAGS) -o CPU CPU.o

clean:
	rm *.o CPU