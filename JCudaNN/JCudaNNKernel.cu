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

// 線形和の計算
extern "C"
__global__ void calc_linear(float *z, float **w, float *xin, int xsize, int ysize) {
  const int y = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (y < ysize) {
    float ztmp = 0.0f;
    for (int x = 0; x < xsize; ++x) {
      ztmp += xin[x] * w[x][y];
    }
    z[y] = ztmp;
  }
}

// 非線形出力関数のベクトル計算
extern "C"
__global__ void calc_output(float *outz, float *z, int ysize, int format) {
  const int y = blockDim.x * blockIdx.x + threadIdx.x;
   
  switch (format) {
  case 0:	// sigmoid
    if (y < ysize) {
      outz[y] = sigmoid(z[y]);
    }
    break;
    
  case 1:	// softmax
    if (y < ysize) {
      float xmax = calc_max(z, ysize);
      float div = calc_div(z, ysize, xmax);
      outz[y] = calc_softmax(z, xmax, div, y);
    }
    break;
  }
}

// 損失関数のベクトル計算
extern "C"
__global__ void loss_derivative(float *out, float *in, float *outz, int outn) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < outn) {
    out[i] = loss_deriv(outz[i], in[i]);
  }
}

// 損失関数の b 方向の微分
extern "C"
__global__ void calc_deriv_b_kernel(float *db, float **w, float *outz, float *in, int xsize, int ysize) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (x < xsize) {
    float d = 0.0f;
    for (int y = 0; y < ysize; ++y) {
      d += w[x][y] * in[y];
    }
    d *= sigmoid_deriv(outz[x]);
    db[x] = d;
  }
}

// 損失関数の w 方向の微分
extern "C"
__global__ void calc_deriv_w_kernel(float **dw, float *in, int xsize, float *db, int ysize) {
  const int x = blockDim.x * blockIdx.x + threadIdx.x;
  const int y = blockDim.y * blockIdx.y + threadIdx.y;
  
  if (x < xsize && y < ysize) {
    dw[x][y] = in[x] * db[y];
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

// １次元のベクトル加算
extern "C"
__global__ void vec_add_1d(float *wout, float *win, int size) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < size) {
    wout[i] += win[i];
  }
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
__global__ void learn_1d(float *wout, float *deriv, float lrate, int size, float nsample) {
  const int i = blockDim.x * blockIdx.x + threadIdx.x;
  
  if (i < size) {
    wout[i] -= lrate * deriv[i] / nsample; 
  }
}
