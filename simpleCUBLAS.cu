#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Includes, cuda */
#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
  if (code != cudaSuccess) 
  {
    fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) exit(code);
	}
}

#define CUBLAS_SAFE_CALL(ans) { cublasAssert((ans), __FILE__, __LINE__); }
inline void cublasAssert(cublasStatus_t status, const char *file, int line, bool abort=true)
{
  if (status != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "!!!! CUBLAS error\n");
    if (abort) exit(EXIT_FAILURE);
  }
}

void flush_l2(cudaStream_t stream) {
  int dev_id = 0;
  int l2_size = 0;
  CUDA_SAFE_CALL(cudaGetDevice(&dev_id));
  CUDA_SAFE_CALL(cudaDeviceGetAttribute(&l2_size, cudaDevAttrL2CacheSize, dev_id));
  printf("Flushing device %d L2 %d bytes\n", dev_id, l2_size);
  void *buffer = 0;
  CUDA_SAFE_CALL(cudaMalloc(&buffer, l2_size));
  CUDA_SAFE_CALL(cudaMemsetAsync(buffer, 0, l2_size, stream));
  CUDA_SAFE_CALL(cudaFree(buffer));
}

#ifdef VALIDATE_OUTPUT
static void simple_sgemm(int n, float alpha, const float *A, const float *B,
                         float beta, float *C) {
  int i;
  int j;
  int k;

  for (i = 0; i < n; ++i) {
    for (j = 0; j < n; ++j) {
      float prod = 0;

      for (k = 0; k < n; ++k) {
        prod += A[k * n + i] * B[j * n + k];
      }

      C[j * n + i] = alpha * prod + beta * C[j * n + i];
    }
  }
}
#endif

# define L2_LINE_SIZE 128
__global__ void prefetch_l2(void *ptr, size_t num_bytes) {
  for (size_t i = 0; i < num_bytes; i+=L2_LINE_SIZE) {
    asm volatile ("prefetch.global.L2::evict_last [%0];" ::"l"((uint8_t*)ptr + i) :);
  }
}


#define M (1024)
#define K (2048)
#define N (1024)
#define MxK M*K
#define KxN K*N
#define MxN M*N
//#define VALIDATE_OUTPUT

/* Main */
int main(int argc, char **argv) {
  bool persistent = true;
  float *host_inp;
  float *host_wgt;
  float *host_outp;
  float *device_inp = 0;
  float *device_wgt = 0;
  float *device_outp = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int i;
#ifdef VALIDATE_OUTPUT
  float *host_outp_ref;
  float error_norm;
  float ref_norm;
  float diff;
#endif
  cublasHandle_t handle;

  /* Initialize CUBLAS */
  size_t inp_mb = sizeof(host_inp[0])*MxK/1024/1024;
  size_t wgt_mb = sizeof(host_wgt[0])*KxN/1024/1024;
  size_t outp_mb = sizeof(host_outp[0])*MxN/1024/1024;
  printf("simpleCUBLAS test running..\n");
  printf("Gemm [%d,%d] x [%d,%d] -> [%d,%d]\n", M,K, K,N, M,N);
  printf("Input Bytes %zuMB\n", inp_mb);
  printf("Weight Bytes %zuMB\n", wgt_mb); 
  printf("Output Bytes %zuMB\n", outp_mb);
  printf("Total Input Bytes: %zuMB\n", inp_mb + wgt_mb);
  printf("Total Device Bytes: %zuMB\n", inp_mb + wgt_mb + outp_mb);

  CUBLAS_SAFE_CALL(cublasCreate(&handle));

  /* Allocate host memory for the matrices */
  host_inp = reinterpret_cast<float *>(malloc(MxK * sizeof(host_inp[0])));
  host_wgt = reinterpret_cast<float *>(malloc(KxN * sizeof(host_wgt[0])));
  host_outp = reinterpret_cast<float *>(malloc(MxN * sizeof(host_outp[0])));

  if (host_inp == 0 || host_wgt == 0 || host_outp == 0) {
    fprintf(stderr, "!!!! host memory allocation error)\n");
    return EXIT_FAILURE;
  }

  /* Fill the matrices with test data */
  for (i = 0; i < MxK; i++) {
    host_inp[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (i = 0; i < KxN; i++) {
    host_wgt[i] = rand() / static_cast<float>(RAND_MAX);
  }
  for (i = 0; i < MxN; i++) {
    host_outp[i] = rand() / static_cast<float>(RAND_MAX);
  }

  /* Allocate device memory for the matrices */
  float *base_addr = device_inp;
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_inp), MxK * sizeof(device_inp[0])))
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_wgt), KxN * sizeof(device_wgt[0])))
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_outp), MxN * sizeof(device_outp[0])))
  printf("Device Pointers: inp=%p, wgt=%p, outp=%p\n", device_inp, device_wgt, device_outp);
  
  /* Verify allocations are contiguous */
  if (!((size_t)device_inp + inp_mb*1024*1024 == (size_t)device_wgt)) {
    printf("Device allocated memory is not contiguous\n");
		return EXIT_FAILURE;
  }
  if (!((size_t)device_wgt + wgt_mb*1024*1024 == (size_t)device_outp)) {
    printf("Device allocated memory is not contiguous\n");
		return EXIT_FAILURE;
  }

  /* Initialize the device matrices with the host matrices */
  CUBLAS_SAFE_CALL(cublasSetVector(MxK, sizeof(host_inp[0]), host_inp, 1, device_inp, 1));
  CUBLAS_SAFE_CALL(cublasSetVector(KxN, sizeof(host_wgt[0]), host_wgt, 1, device_wgt, 1));
  CUBLAS_SAFE_CALL(cublasSetVector(MxN, sizeof(host_outp[0]), host_outp, 1, device_outp, 1));

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  cudaStreamAttrValue stream_attribute;

  if (persistent) {
    size_t alloc_num_bytes = 32*1024*1024; // Allocate 32MB
    int device_id = 0;

    // Set-aside 3/4 of L2 cache for persisting accesses or the max allowed
    cudaDeviceProp prop;
    CUDA_SAFE_CALL(cudaGetDeviceProperties( &prop, device_id));
    size_t size = std::min<size_t>( int(prop.l2CacheSize * 0.75) , prop.persistingL2CacheMaxSize );
    CUDA_SAFE_CALL(cudaDeviceSetLimit( cudaLimitPersistingL2CacheSize, size));

    // Set persistent memory size
    size_t window_size = std::min<size_t>(prop.accessPolicyMaxWindowSize, alloc_num_bytes);
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(base_addr);
    stream_attribute.accessPolicyWindow.num_bytes = window_size;
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    CUDA_SAFE_CALL(
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute)
    );
    std::cout << "Allocate Persistent L2: " << window_size << " B" << std::endl;
    std::cout << "Max Window Size: " << prop.accessPolicyMaxWindowSize << std::endl;
  }

#ifdef VALIDATE_OUTPUT
  /* Performs operation using plain C code */
  printf("Run reference Gemm\n");
  simple_sgemm(N, alpha, host_inp, host_wgt, beta, host_outp);
  host_outp_ref = host_outp;
#endif

  /* Performs operation using cublas */
  printf("Warmup\n");
  flush_l2(stream);
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));

	printf("Test flush\n");
  flush_l2(stream);
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));
  flush_l2(stream);
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));


  printf("Test prefetch kernel\n");
  flush_l2(stream);
	prefetch_l2<<<1,1>>>(base_addr, inp_mb*1024*1024+wgt_mb*1024*1024);
  CUBLAS_SAFE_CALL(
      cublasSgemm(
        handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, 
        device_inp, N, 
        device_wgt, N, 
        &beta, 
        device_outp, N
  ));

  /* Allocate host memory for reading back the result from device memory */
  host_outp = reinterpret_cast<float *>(malloc(MxN * sizeof(host_outp[0])));

  if (host_outp == 0) {
    fprintf(stderr, "!!!! host memory allocation error (C)\n");
    return EXIT_FAILURE;
  }

  /* Read the result back */
  CUBLAS_SAFE_CALL(cublasGetVector(MxN, sizeof(host_outp[0]), device_outp, 1, host_outp, 1));

#ifdef VALIDATE_OUTPUT
  /* Check result against reference */
  error_norm = 0;
  ref_norm = 0;

  for (i = 0; i < MxN; ++i) {
    diff = host_outp_ref[i] - host_outp[i];
    error_norm += diff * diff;
    ref_norm += host_outp_ref[i] * host_outp_ref[i];
  }

  error_norm = static_cast<float>(sqrt(static_cast<double>(error_norm)));
  ref_norm = static_cast<float>(sqrt(static_cast<double>(ref_norm)));

  if (fabs(ref_norm) < 1e-7) {
    fprintf(stderr, "!!!! reference norm is 0\n");
    return EXIT_FAILURE;
  }
  free(host_outp_ref);
#endif

  /* Memory clean up */
  free(host_inp);
  free(host_wgt);
  free(host_outp);
  CUDA_SAFE_CALL(cudaFree(device_inp));
  CUDA_SAFE_CALL(cudaFree(device_wgt));
  CUDA_SAFE_CALL(cudaFree(device_outp));
  CUBLAS_SAFE_CALL(cublasDestroy(handle));

  // Remove any persistent lines in L2
  if (persistent) {
    stream_attribute.accessPolicyWindow.num_bytes = 0;
    CUDA_SAFE_CALL(
      cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute)
    );
    CUDA_SAFE_CALL(cudaCtxResetPersistingL2Cache());
  }

#ifdef VALIDATE_OUTPUT
  if (error_norm / ref_norm < 1e-6f) {
    printf("simpleCUBLAS test passed.\n");
    exit(EXIT_SUCCESS);
  } else {
    printf("simpleCUBLAS test failed.\n");
    exit(EXIT_FAILURE);
  }
#endif
}
