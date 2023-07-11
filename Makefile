CC = nvcc
CFLAGS = -O3 -std=c++11
LDFLAGS = -lcublas

simpleCUBLAS: simpleCUBLAS.cpp
		$(CC) $(CFLAGS) $(LDFLAGS) -o simpleCUBLAS simpleCUBLAS.cpp

clean:
		rm -f simpleCUBLAS
