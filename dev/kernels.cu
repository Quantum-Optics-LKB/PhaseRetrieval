//Kernels for the WISH routines
// Compiler command to test compile using nvcc :
// nvcc -gencode=arch=compute_61,code=sm_61 --use_fast_math -I=/home/tangui/anaconda3/lib/python3.8/site-packages/cupy/_core/include,/home/tangui/anaconda3/lib/python3.8/site-packages/cupy/_core/include/cupy/_cuda/cuda-11.4,/usr/local/cuda/include -ftz=true --cubin -o=kernels kernels.cu
//block = (8, 512, 512)
//grid = (SLM.shape[0]//block[0], SLM.shape[1]//block[1], SLM.shape[2]//block[2])
#include <cupy/complex.cuh>
#include <cufft.h>
#include <stdio.h>

extern "C"{
  __global__ void impose_amp(const complex<float>* y, const complex<float>* x, complex<float>* out, const int N0, const int N1, const int N2){

    int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = blockDim.y * blockIdx.y + threadIdx.y;
    int i2 = blockDim.z * blockIdx.z + threadIdx.z;
    int tid = N1*N2*i0 + N2*i1 + i2;
    // //prevent illegal memory access
    if (i0 < N0 && i1 < N1 && i2 < N2){
        
    float ang = arg(x[tid]);
    out[tid] = abs(y[tid]) * exp(complex<float>(0.0f, 1.0f) * ang);
    
    }
  }

  __global__ void multiply_conjugate(const complex<float>* y, complex<float>* x, const int N0, const int N1, const int N2){

    // int tid =  blockDim.x * blockIdx.x + threadIdx.x;
    int i0 = blockDim.x * blockIdx.x + threadIdx.x;
    int i1 = blockDim.y * blockIdx.y + threadIdx.y;
    int i2 = blockDim.z * blockIdx.z + threadIdx.z;
    int tid = N1*N2*i0 + N2*i1 + i2;
    
    if (i0 < N0 && i1 < N1 && i2 < N2){
    
      x[tid] *= conj(y[tid]);
    
    }
  }
  //pass .handle attr of cupy fft_plan (int ptr)
  // pass A0.data.ptr data pointer from python side
  // __global__ void frt_gpu_vec_s(thrust::device_vector<thrust::complex<float>>& A0, const float d1x, const float d1y,
  //   const float wv, const float z, const cufftHandle plan){

  //   // int tid = blockDim.x * blockIdx.x + threadIdx.x;
  //   // auto _A0 = const_cast<cufftComplex*>(reinterpret_cast<const cufftComplex*>(thrust::raw_pointer_cast(A0.data())));
  //   // if (z>0){
  //   //   cufftExecC2C(plan, _A0, _A0, CUFFT_FORWARD);
  //   //   A0[tid].x *= d1x * d1y;
  //   //   A0[tid].y *= d1x * d1y;
  //   // }
  //   // else{
  //   //   cufftExecC2C(plan, _A0, _A0, CUFFT_INVERSE);
  //   //   A0[tid].x *= d1x * d1y;
  //   //   A0[tid].y *= d1x * d1y;
  //   // }
    
  //   // A0[tid] = A0[tid] / (complex<float>(0.0f, 1.0f) * wv * z);
  //   printf(typeid(A0).name());

  // }
}
