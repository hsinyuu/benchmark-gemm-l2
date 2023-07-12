CC = nvcc
CFLAGS = -O3 -std=c++11
LDFLAGS = -lcublas

simpleCUBLAS: simpleCUBLAS.cu
		$(CC) $(CFLAGS) $(LDFLAGS) -o simpleCUBLAS simpleCUBLAS.cu -arch=sm_80

ptx: simpleCUBLAS.cu
		$(CC) $(CFLAGS) $(LDFLAGS) simpleCUBLAS.cu -arch=sm_80 -ptx

clean:
		rm -f simpleCUBLAS
