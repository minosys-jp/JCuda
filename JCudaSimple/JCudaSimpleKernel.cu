// JCudaSimpleKernel.java から呼び出される CUDA カーネル関数
extern "C"
__global__ void simpleKernel(float** input, int xsize, int ysize, float* output)
{
  // X 方向については並列化されている
  const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (x < xsize) {
    // Y 方向については逐次足し算する
    // そのため、メモリアクセスの競合は考えなくても良い
    for (int y = 0; y < ysize; ++y) {
      output[x] += input[x][y];
    }
  }
  __syncthreads();
}
