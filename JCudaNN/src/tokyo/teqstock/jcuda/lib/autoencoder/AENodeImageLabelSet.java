package tokyo.teqstock.jcuda.lib.autoencoder;

import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

/**
 * リーフノードの情報蓄積クラス
 * 
 * @author minoru
 *
 */
public class AENodeImageLabelSet extends AEImageLabelSet {
	private static final int NTHREAD2 = 8;
	private final Map<String, CUfunction> fMapper;
	private final int lc;
	CUdeviceptr devOutLabel;
	CUdeviceptr[] devOutLabelArray;
	
	private int calcBlock2D(int size) {
		return (size + NTHREAD2 - 1) / NTHREAD2;
	}
	
	/**
	 * constructor
	 * 
	 * @param n
	 * @param wh
	 * @param lc
	 */
	public AENodeImageLabelSet(Map<String, CUfunction> fMapper, int n, int wh, int lc) {
		super(n, wh);
		// TODO Auto-generated constructor stub
		this.fMapper = fMapper;
		this.lc = lc;
	}

	public void finalize() {
		if (devOut != null) {
			IntStream.range(0, n).forEach(i->{
				cuMemFree(devOutArray[i]);
			});
			cuMemFree(devOut);
			devOut = null;
		}
		if (devOutLabel != null) {
			IntStream.range(0, n).forEach(i->{
				cuMemFree(devOutLabelArray[i]);
			});
			cuMemFree(devOutLabel);
			devOutLabel = null;
		}
	}
	
	/**
	 * ノード情報をコピーする
	 * 
	 * @param devImage
	 * @param devLabel
	 */
	public void setContentDev(CUdeviceptr devImage, CUdeviceptr devLabel) {
		if (devOut == null) {
			// メモリを確保する
			devOutArray = new CUdeviceptr[n];
			IntStream.range(0, n).forEach(i->{
				devOutArray[i] = new CUdeviceptr();
				cuMemAlloc(devOutArray[i], Sizeof.FLOAT * wh);
			});
			devOut = new CUdeviceptr();
			cuMemAlloc(devOut, Sizeof.POINTER * n);
			cuMemcpyHtoD(devOut, Pointer.to(devOutArray), Sizeof.POINTER * n);
			
			// AEImage の作成
			image = new AEImageLabelSet.AEImage(devOut, devOutArray, wh);
		}
		if (devOutLabel == null) {
			// メモリを確保する
			devOutLabelArray = new CUdeviceptr[n];
			IntStream.range(0, n).forEach(i->{
				devOutLabelArray[i] = new CUdeviceptr();
				cuMemAlloc(devOutLabelArray[i], Sizeof.FLOAT * lc);
			});
			devOutLabel = new CUdeviceptr();
			cuMemAlloc(devOutLabel, Sizeof.POINTER * n);
			cuMemcpyHtoD(devOutLabel, Pointer.to(devOutLabelArray), Sizeof.POINTER * n);
			
			// AELabel の作成
			label = new AEImageLabelSet.AELabel(devOutLabel, devOutLabelArray, lc);
		}
		
		// イメージ、教師データを作成する
		Pointer kp = Pointer.to(
				Pointer.to(devOut),
				Pointer.to(devImage),
				Pointer.to(new int[]{n}),
				Pointer.to(new int[]{wh})
				);
		cuLaunchKernel(fMapper.get("copy2D"),
				calcBlock2D(n), calcBlock2D(wh), 1,
				NTHREAD2, NTHREAD2, 1,
				0, null,
				kp, null	
				);
		kp = Pointer.to(
				Pointer.to(devOutLabel),
				Pointer.to(devLabel),
				Pointer.to(new int[]{n}),
				Pointer.to(new int[]{lc})
				);
		cuLaunchKernel(fMapper.get("copy2D"),
				calcBlock2D(n), calcBlock2D(lc), 1,
				NTHREAD2, NTHREAD2, 1,
				0, null,
				kp, null
				);
		cuCtxSynchronize();
	}
}
