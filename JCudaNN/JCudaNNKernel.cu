// JCudaNN から呼び出される CUDA カーネル関数

// *** デバイス定義の関数 ***

// 損失関数の微分
__device__
static float loss_deriv(float x, float y) {
  return x - y;
}

// sigmoid 関数
__device__
static float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

// sigmoid 関数の微分
__device__
static float sigmoid_deriv(float outz) {
  return outz * (1.0f - outz);
}

// 最大値を求める
__device__
static float calc_max(float *xn, int n) {
  // 最大値を求める
  float xmax = -1e+5;
  for (int i = 0; i < n; ++i) {
    if (xmax < xn[i]) {
      xmax = xn[i];
    }
  }
  return xmax;
}

// softmax 関数の分母を求める
__device__
static float calc_div(float *xn, int n, float xmax) {
  float div = 0.0f;
  for (int i = 0; i < n; ++i) {
    div += expf(xn[i] - xmax);
  }
  return div;
}

// softmax 関数
__device__
static float calc_softmax(float *xn, float xmax, float div, int m) {
  return expf(xn[m] - xmax) / div;
}

// *** SimpleNet.java からの呼び出し ***

// １次元クリア
extern "C"
__global__ void clear1D(float *b, int n) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  if (x < n) {
    b[x] = 0.0f;
  }
}

// ２次元クリア
extern "C"
__global__ void clear2D(float **w, int xsize, int ysize) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && y < ysize) {
    w[x][y] = 0.0f;
  } 
}

// forward 演算
extern "C"
__global__ void calc_forward(float **outz, float **tmpz, float **w, float **xin, int xsize, int ysize, int fmt, int bs) {
  const int y = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.y * blockIdx.y + threadIdx.y;

  // 線形和の計算   
  if (y < ysize && k < bs) {
    float ztmp = 0.0f;
    for (int x = 0; x < xsize; ++x) {
      ztmp += xin[k][x] * w[x][y];
    }
    tmpz[k][y] = ztmp;
  }
  __syncthreads();
  
  // 非線形関数の出力
  if (y < ysize && k < bs) {
    if (fmt == 1) {
      // softmax 関数
      float xmax = calc_max(tmpz[k], ysize);
      float div = calc_div(tmpz[k], ysize, xmax);
      outz[k][y] = calc_softmax(tmpz[k], xmax, div, y); 
    } else {
      // sigmoid 関数
      outz[k][y] = sigmoid(tmpz[k][y]);
    }
  }
}

// 損失関数のベクトル計算
extern "C"
__global__ void loss_derivative(float **out, float **outz, float **label, int *samples, int outn, int bs) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (i < outn && k < bs) {
    out[k][i] = loss_deriv(outz[k][i], label[samples[k]][i]);
  }
}

// 損失関数の b 方向の微分
extern "C"
__global__ void calc_deriv_b_kernel(float **db, float **w, float **outz, float **bderiv2,
  int xsize, int ysize, int bs) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int k = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && k < bs) {
    float d = 0.0f;
    for (int y = 0; y < ysize; ++y) {
      d += w[x][y] * bderiv2[k][y];
    }
    d *= sigmoid_deriv(outz[k][x]);
    db[k][x] = d;
  }
}

// 損失関数の w 方向の微分
extern "C"
__global__ void calc_deriv_w_kernel(float **dw, float **in, int xsize, float **db,
  int ysize, int bs) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && y < ysize) {
    float d = 0.0f;
    for (int z = 0; z < bs; ++z) {
      d += in[z][x] * db[z][y];
    }
    dw[x][y] = d;
  }
}

// *** NeuralNet.java からの呼び出し ***

// ２次元のベクトル加算
extern "C"
__global__ void vec_add_2d(float **wout, float **win, int xsize, int ysize) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && y < ysize) {
    wout[x][y] += win[x][y];
  }
}

// １次元のベクトル加算(未使用)
extern "C"
__global__ void vec_add_1d(float *bout, float *bin, int size, int bs) {
}

// w に関する学習
extern "C"
__global__ void learn_2d(float **wout, float **deriv, float lrate, int xsize, int ysize, float nsample) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && y < ysize) {
    wout[x][y] -= lrate * deriv[x][y] / nsample;
  }
}

// b に関する学習
extern "C"
__global__ void learn_1d(float *bout, float **deriv, float lrate, int size, float nsample, int bs) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < size) {
    for (int s = 0; s < bs; ++s) {
      bout[i] -= lrate * deriv[s][i] / nsample; 
    }
  }
}

// テスト用
extern "C"
__global__ void test_sum(float *dout, float **dw, float **db, int inn, int outnn, int bs) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x; // for out

  // zero clear
  dout[i] = 0.0f;
  
  // for input
  for (int k = 0; k < inn; ++k) {
    dout[i] += fabsf(dw[k][i]);
  }
  __syncthreads();
  
  // for batchsize
  for (int k = 0; k < bs; ++k) {
    dout[i] += fabsf(db[k][i]);
  }
}
