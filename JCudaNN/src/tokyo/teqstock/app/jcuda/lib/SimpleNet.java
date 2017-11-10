package tokyo.teqstock.app.jcuda.lib;

import java.io.IOException;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

/**
 * simple neural network
 * @author minoru
 *
 */
public class SimpleNet {
	private static final float EPS = 1.000f;
	private static final int NTHREAD = 32;	// for 1D
	private static final int NTHREAD2 = 8;	// for 2D
	private final Map<String, CUfunction> fMapper;
	
	/**
	 * input terminal quantity
	 */
	private final int inn;
	public int getInn() { return inn; }

	/**
	 * output terminal quantity
	 */
	private final int outn;
	public int getOutn() { return outn; }

	
	/**
	 * weight matrix
	 */
	public CUdeviceptr[] devWArray;
	public CUdeviceptr devW;
	
	/**
	 * bias
	 */
	public CUdeviceptr devB;
	public OutputFormat format;

	/**
	 * forward output caches
	 */
	public CUdeviceptr devOutz;
	
	/**
	 * forward output caches 2D
	 */
	public CUdeviceptr[] devOutz2D;

	private CUdeviceptr devTmpz;
	private CUdeviceptr[] devTmpz2D;
	
	/**
	 * batch size
	 */
	private final int batchsize;
	
	/**
	 * output format
	 * @author minoru
	 *
	 */
	enum OutputFormat {
		SIGMOID, SOFTMAX
	};

	private int calcBlock(int size) {
		return (size + NTHREAD - 1) / NTHREAD;
	}
	
	private int calcBlock2D(int size) {
		return (size + NTHREAD2 - 1) / NTHREAD2;
	}
	
	/**
	 * calculate the derivative for the biases of the loss function
	 * @param outer
	 * @param w
	 * @param a
	 * @param z
	 * @return
	 */
	public void calc_deriv_b(CUdeviceptr devDB, CUdeviceptr outz, CUdeviceptr in, boolean bMostouter) {
		if (!bMostouter) {
			Pointer kp = Pointer.to(
					Pointer.to(devDB), Pointer.to(devW), Pointer.to(outz), Pointer.to(in),
					Pointer.to(new int[]{inn}), Pointer.to(new int[]{outn}),
					Pointer.to(new int[]{batchsize})
					);
			// 隠れ層の計算
			cuLaunchKernel(fMapper.get("calc_deriv_b_kernel"),
					calcBlock2D(inn), calcBlock2D(batchsize), 1,
					NTHREAD2, NTHREAD2, 1,
					0, null,
					kp, null
					);
		} else {
			Pointer kp = Pointer.to(Pointer.to(devDB),
					Pointer.to(devOutz),	// cached forward values
					Pointer.to(outz),		// top of bHot
					Pointer.to(in),			// samples
					Pointer.to(new int[]{outn}), Pointer.to(new int[]{batchsize}));
			// 最外殻では損失関数の微分を通す
			cuLaunchKernel(fMapper.get("loss_derivative"),
					calcBlock2D(outn), calcBlock2D(batchsize), 1,
					NTHREAD2, NTHREAD2, 1,
					0, null,
					kp, null
					);
		}
		cuCtxSynchronize();
	}

	/**
	 * calculate derivative for the weights
	 * @param a
	 * @param delta
	 * @return
	 */
	public void calc_deriv_w(CUdeviceptr devDW, 
			CUdeviceptr in, int xsize, CUdeviceptr delta, int ysize) {
		Pointer kp = Pointer.to(Pointer.to(devDW),
				Pointer.to(in), Pointer.to(new int[]{xsize}),
				Pointer.to(delta), Pointer.to(new int[]{ysize}),
				Pointer.to(new int[]{batchsize})
		);
		cuLaunchKernel(fMapper.get("calc_deriv_w_kernel"),
				calcBlock2D(xsize), calcBlock2D(ysize), 1,
				NTHREAD2, NTHREAD2, 1,
				0, null,
				kp, null);
		cuCtxSynchronize();
	}

	/**
	 * constructor
	 * @param inn
	 * @param outn
	 */
	public SimpleNet(Map<String, CUfunction> fMapper, int inn, int outn, int batchsize) {
		this.inn = inn;
		this.outn = outn;
		this.batchsize = batchsize;
		this.fMapper = fMapper;
		devWArray = new CUdeviceptr[inn];
		IntStream.range(0,  inn).forEach(i->{
			devWArray[i] = new CUdeviceptr();
			cuMemAlloc(devWArray[i], Sizeof.FLOAT * outn);
			float[] tmpw = new float[outn];
			IntStream.range(0,  outn).forEach(j->{
				tmpw[j] = (float)(2.0 * Math.random() - 1.0) * EPS;
			});
			cuMemcpyHtoD(devWArray[i], Pointer.to(tmpw), Sizeof.FLOAT * outn);
		});
		devW = new CUdeviceptr();
		cuMemAlloc(devW, Sizeof.POINTER * inn);
		cuMemcpyHtoD(devW, Pointer.to(devWArray), Sizeof.POINTER * inn);
		devB = new CUdeviceptr();
		cuMemAlloc(devB, Sizeof.FLOAT * outn);
		devOutz2D = new CUdeviceptr[batchsize];
		IntStream.range(0, batchsize).forEach(i->{
			devOutz2D[i] = new CUdeviceptr();
			cuMemAlloc(devOutz2D[i], Sizeof.FLOAT * outn);
		});
		devOutz = new CUdeviceptr();
		cuMemAlloc(devOutz, Sizeof.POINTER * batchsize);
		cuMemcpyHtoD(devOutz, Pointer.to(devOutz2D), Sizeof.POINTER * batchsize);
		devTmpz2D = new CUdeviceptr[batchsize];
		IntStream.range(0, batchsize).forEach(i->{
			devTmpz2D[i] = new CUdeviceptr();
			cuMemAlloc(devTmpz2D[i], Sizeof.FLOAT * outn);
		});
		devTmpz = new CUdeviceptr();
		cuMemAlloc(devTmpz, Sizeof.POINTER * batchsize);
		cuMemcpyHtoD(devTmpz, Pointer.to(devTmpz2D), Sizeof.POINTER * batchsize);
		format = OutputFormat.SIGMOID;
	}

	public void finalize() {
		IntStream.range(0, inn).forEach(i->{
			cuMemFree(devWArray[i]);
		});
		cuMemFree(devW);	
		cuMemFree(devB);
		IntStream.range(0, batchsize).forEach(i->{
			cuMemFree(devOutz2D[i]);
		});
		cuMemFree(devOutz);
		IntStream.range(0, batchsize).forEach(i->{
			cuMemFree(devTmpz2D[i]);
		});
		cuMemFree(devTmpz);
	}
	
	/**
	 * forward operation
	 * @param in
	 * @return
	 */
	public CUdeviceptr forward(CUdeviceptr in) {
		// format を format 番号に変換
		int fmt = 0;
		switch (format) {
		case SIGMOID:
			fmt = 0;
			break;
		case SOFTMAX:
			fmt = 1;
			break;
		}

		// foward 計算
		Pointer kp = Pointer.to(Pointer.to(devOutz), Pointer.to(devTmpz), 
				Pointer.to(devW), Pointer.to(in),
				Pointer.to(new int[]{inn}), Pointer.to(new int[]{outn}),
				Pointer.to(new int[]{fmt}), Pointer.to(new int[]{batchsize})
				);
		cuLaunchKernel(fMapper.get("calc_forward"),
				calcBlock2D(outn), calcBlock2D(batchsize), 1,
				NTHREAD2, NTHREAD2, 1,
				0, null,
				kp, null
				);
		cuCtxSynchronize();
		return devOutz;
	}
}
