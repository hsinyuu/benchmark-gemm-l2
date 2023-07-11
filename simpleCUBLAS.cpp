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

#define M (2048)
#define K (2048)
#define N (2048)
#define MxK M*K
#define KxN K*N
#define MxN M*N
//#define VALIDATE_OUTPUT

/* Main */
int main(int argc, char **argv) {
  cublasStatus_t status;
  std::cout<<MxK<<std::endl;
  float *host_inp;
  float *host_wgt;
  float *host_outp;
  float *host_outp_ref;
  float *device_inp = 0;
  float *device_wgt = 0;
  float *device_outp = 0;
  float alpha = 1.0f;
  float beta = 0.0f;
  int i;
  float error_norm;
  float ref_norm;
  float diff;
  cublasHandle_t handle;

  /* Initialize CUBLAS */
  printf("simpleCUBLAS test running..\n");
  printf("Gemm [%d,%d] x [%d,%d] -> [%d,%d]\n", M,K, K,N, M,N);
  printf("Input Bytes %zuMB\n", sizeof(host_inp[0])*MxK/1024/1024);
  printf("Weight Bytes %zuMB\n", sizeof(host_wgt[0])*KxN/1024/1024);
  printf("Output Bytes %zuMB\n", sizeof(host_outp[0])*MxN/1024/1024);
  printf("Total Input Bytes: %zuMB\n", (sizeof(host_inp[0])*MxK + sizeof(host_wgt[0])*KxN)/1024/1024);
  printf("Total Output Bytes: %zuMB\n", sizeof(host_outp[0])*MxN/1024/1024);

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
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_inp), MxK * sizeof(device_inp[0])))
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_wgt), KxN * sizeof(device_wgt[0])))
  CUDA_SAFE_CALL(cudaMalloc(reinterpret_cast<void **>(&device_outp), MxN * sizeof(device_outp[0])))

  /* Initialize the device matrices with the host matrices */
  CUBLAS_SAFE_CALL(cublasSetVector(MxK, sizeof(host_inp[0]), host_inp, 1, device_inp, 1));
  CUBLAS_SAFE_CALL(cublasSetVector(KxN, sizeof(host_wgt[0]), host_wgt, 1, device_wgt, 1));
  CUBLAS_SAFE_CALL(cublasSetVector(MxN, sizeof(host_outp[0]), host_outp, 1, device_outp, 1));


#ifdef VALIDATE_OUTPUT
  /* Performs operation using plain C code */
  printf("Run reference Gemm\n");
  simple_sgemm(N, alpha, host_inp, host_wgt, beta, host_outp);
  host_outp_ref = host_outp;
#endif

  /* Performs operation using cublas */
  printf("Run Cublas Gemm\n");
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
#endif

  /* Memory clean up */
  free(host_inp);
  free(host_wgt);
  free(host_outp);
  free(host_outp_ref);
  CUDA_SAFE_CALL(cudaFree(device_inp));
  CUDA_SAFE_CALL(cudaFree(device_wgt));
  CUDA_SAFE_CALL(cudaFree(device_outp));

  CUBLAS_SAFE_CALL(cublasDestroy(handle));

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
