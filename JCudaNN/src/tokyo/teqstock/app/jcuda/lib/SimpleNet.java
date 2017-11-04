package tokyo.teqstock.app.jcuda.lib;

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
	private static final float EPS = 0.01f;
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
	 * linear part
	 */
	public CUdeviceptr devZ;

	/**
	 * forward output caches
	 */
	public CUdeviceptr devOutz;

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
					Pointer.to(new int[]{inn}), Pointer.to(new int[]{outn})
					);
			// 隠れ層の計算
			cuLaunchKernel(fMapper.get("calc_deriv_b_kernel"),
					calcBlock(inn), 1, 1,
					NTHREAD, 1, 1,
					0, null,
					kp, null
					);
		} else {
			Pointer kp = Pointer.to(Pointer.to(devDB), Pointer.to(in), Pointer.to(devOutz),
					Pointer.to(new int[]{outn}));
			// 最外殻では損失関数の微分を通す
			cuLaunchKernel(fMapper.get("loss_derivative"),
					calcBlock(outn), 1, 1,
					NTHREAD, 1, 1,
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
				Pointer.to(delta), Pointer.to(new int[]{ysize})
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
	public SimpleNet(Map<String, CUfunction> fMapper, int inn, int outn) {
		this.inn = inn;
		this.outn = outn;
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
		devZ = new CUdeviceptr();
		cuMemAlloc(devZ, Sizeof.FLOAT * outn);
		devOutz = new CUdeviceptr();
		cuMemAlloc(devOutz, Sizeof.FLOAT * outn);
		Pointer kp = Pointer.to(Pointer.to(devB), Pointer.to(new int[]{outn}));
		cuLaunchKernel(fMapper.get("clear1D"),
				calcBlock(outn), 1, 1,
				NTHREAD, 1, 1,
				0, null,
				kp, null);
		cuCtxSynchronize();
		format = OutputFormat.SIGMOID;
	}

	/**
	 * forward operation
	 * @param in
	 * @return
	 */
	public CUdeviceptr forward(CUdeviceptr in) {
		// z をクリアする
		Pointer kp = Pointer.to(Pointer.to(devZ), Pointer.to(new int[]{outn}));
		cuLaunchKernel(fMapper.get("clear1D"),
				calcBlock(outn), 1, 1,
				NTHREAD, 1, 1,
				0, null,
				kp, null
				);
		cuCtxSynchronize();
		
		// 線形和を計算する
		kp = Pointer.to(Pointer.to(devZ), Pointer.to(devW), Pointer.to(in),
				Pointer.to(new int[]{inn}), Pointer.to(new int[]{outn})
				);
		cuLaunchKernel(fMapper.get("calc_linear"),
				calcBlock(outn), 1, 1,
				NTHREAD, 1, 1,
				0, null,
				kp, null
				);
		cuCtxSynchronize();
		
		// 非線形関数を通す
		int fmt = 0;
		switch (format) {
		case SIGMOID:
			fmt = 0;
			break;
		case SOFTMAX:
			fmt = 1;
			break;
		}
		kp = Pointer.to(Pointer.to(devOutz), Pointer.to(devZ),
				Pointer.to(new int[] {outn}), Pointer.to(new int[]{fmt}));
		cuLaunchKernel(fMapper.get("calc_output"),
				calcBlock(outn), 1, 1,
				NTHREAD, 1, 1,
				0, null,
				kp, null
				);
		cuCtxSynchronize();
		return devOutz;
	}
	
	/**
	 * copy devOutz to the host memory
	 * @return host memory
	 */
	public float[] getOutz() {
		float[] host = new float[outn];
		cuMemcpyDtoH(Pointer.to(host), devOutz, Sizeof.FLOAT * outn);
		return host;
	}
}
