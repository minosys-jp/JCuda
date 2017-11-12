package tokyo.teqstock.jcuda.lib.autoencoder;

import java.io.IOException;
import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;
import static jcuda.jcurand.JCurand.curandCreateGenerator;
import static jcuda.jcurand.JCurand.curandSetPseudoRandomGeneratorSeed;
import static jcuda.jcurand.JCurand.curandGenerateUniform;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.jcurand.curandGenerator;
import tokyo.teqstock.jcuda.lib.NeuralNet;
import tokyo.teqstock.jcuda.lib.SimpleNet;

/**
 * AutoEncoder の実装
 * 
 * @author minoru
 *
 */
public class AutoEncoderNode {
	private static final int CURAND_RNG_PSEUDO_DEFAULT = 100;
	private static final int NTHREAD2 = 8;
	
	private final int n, wha, whb;
	private final AENodeImageLabelSet ils;
	private final NeuralNet nn;
	private final NeuralNet.CUDARegion region;
	private final int[] totology;
	private CUdeviceptr devInNoise;
	private CUdeviceptr[] devInNoiseArray;
	private final curandGenerator generator;
	private CUdeviceptr devNoise;
	private final Map<String, CUfunction> fMapper;
	
	private int calcBlock2D(int size) {
		return (size + NTHREAD2 - 1) / NTHREAD2;
	}
	
	/**
	 * デバイス乱数シミュレータの初期化
	 * 
	 * @return
	 */
	public static curandGenerator createGenerator() {
		curandGenerator generator = new curandGenerator();
		curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
		curandSetPseudoRandomGeneratorSeed(generator, System.currentTimeMillis());
		return generator;
	}
	
	/**
	 * constructor
	 * 
	 * @param fMapper
	 * @param lRate
	 * @param n
	 * @param wha
	 * @param whb
	 * @throws IOException
	 */
	public AutoEncoderNode(Map<String, CUfunction> fMapper, curandGenerator generator,
			float lRate, int n, int wha, int whb) throws IOException {
		this.n = n;
		this.wha = wha;
		this.whb = whb;
		this.fMapper = fMapper; 
		ils = new AENodeImageLabelSet(fMapper, n, wha, wha);
		int[] nodes = new int[]{wha, whb, wha};
		nn = new NeuralNet(fMapper, lRate, nodes, n);
		region = nn.createDefaultCUDARegion();
		totology = IntStream.range(0, n).toArray();
		
		// cuRAND の初期化
		devNoise = new CUdeviceptr();
		cuMemAlloc(devNoise, Sizeof.FLOAT * n * wha);
		this.generator = generator;
	}
	
	public void finalize() {
		cuMemFree(devNoise);
		
		if (devInNoise != null) {
			IntStream.range(0, n).forEach(i->{
				cuMemFree(devInNoiseArray[i]);
			});
			cuMemFree(devInNoise);
		}
	}
	
	/**
	 * contents を設定する; backPropagate の前に呼び出す
	 * @param devIn
	 * @param threshold
	 */
	public void setContentDev(CUdeviceptr devIn, float threshold) {
		if (threshold >= 1.0f) {
			ils.setContentDev(devIn, devIn);
		} else {
			if (devInNoise == null) {
				// 新しくノイズが乗った画像を収容する空間を作る
				devInNoiseArray = new CUdeviceptr[n];
				IntStream.range(0, n).forEach(i->{
					devInNoiseArray[i] = new CUdeviceptr();
					cuMemAlloc(devInNoiseArray[i], Sizeof.FLOAT * wha);
				});
				devInNoise = new CUdeviceptr();
				cuMemAlloc(devInNoise, Sizeof.POINTER * n);
				cuMemcpyHtoD(devInNoise, Pointer.to(devInNoiseArray), Sizeof.POINTER * n);
			}
			
			// [0.0, 1.0) の乱数を作成する
			curandGenerateUniform(generator, devNoise, n * wha);
			
			// ノイズシェーピングする
			Pointer kp = Pointer.to(
					Pointer.to(devInNoise),
					Pointer.to(devIn),
					Pointer.to(devNoise),
					Pointer.to(new int[]{n}),
					Pointer.to(new int[]{wha}),
					Pointer.to(new float[]{threshold})
					);
			cuLaunchKernel(fMapper.get("noise_shape"),
					calcBlock2D(n), calcBlock2D(wha), 1,
					NTHREAD2, NTHREAD2, 1,
					0, null,
					kp, null);
			cuCtxSynchronize();
			ils.setContentDev(devInNoise, devIn);
		}
	}
	
	/**
	 * 逆誤差散乱法
	 * 
	 * @param nset	繰り返し回数
	 */
	public void backPropagate(int nset) {
		IntStream.range(0, nset).forEach(i->{
			nn.backPropagate(region, ils, totology);
		});
	}
	
	/**
	 * ニューラルネットの中間出力（デバイスメモリ）を返す
	 * @return
	 */
	public CUdeviceptr getOut0() {
		return nn.neurons[0].devOutz;
	}
	
	/**
	 * AutoEncoder を適用した結果得られた係数をコピーする
	 * @param sn	コピー先の SimpleNet
	 */
	public void copyCoef(SimpleNet sn) {
		nn.neurons[0].copyCoef(sn);
	}
}
