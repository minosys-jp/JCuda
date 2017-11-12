package tokyo.teqstock.jcuda.lib.autoencoder;

import java.io.IOException;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import jcuda.jcurand.curandGenerator;
import tokyo.teqstock.jcuda.lib.ImageLabelSet;
import tokyo.teqstock.jcuda.lib.NeuralNet;
import tokyo.teqstock.jcuda.lib.SimpleNet;

/**
 * AutoEncoder 実装
 * 
 * @author minoru
 *
 */
public class AutoEncoder {
	private final int batchsize;
	private final int nset;
	private final int nsamples1, nsamples2;
	private final float threshold;
	private final int[] nodes;
	public NeuralNet nn, nnleaf;
	AutoEncoderNode[] aenodes;
	curandGenerator generator;
	CUdeviceptr devIn;
	CUdeviceptr[] devInArray;
	AENodeImageLabelSet ilsleaf;
	NeuralNet.CUDARegion region, regionleaf;
	Random rand;
	
	/**
	 * constructor
	 * 
	 * @param fMapper
	 * @param fParamMap
	 * @param nodes
	 * @param lRate
	 * @param threshold
	 * @throws IOException
	 */
	public AutoEncoder(Map<String, CUfunction> fMapper,
			Map<String, Integer> fParamMap, 
			int[] nodes, float lRate, float threshold) throws IOException {
		this.threshold = threshold;
		this.nodes = nodes;
		this.batchsize = fParamMap.get("BATCHSIZE");
		this.nset = fParamMap.get("NSET");
		this.nsamples1 = fParamMap.get("NSAMPLE1");
		this.nsamples2 = fParamMap.get("NSAMPLE2");
		this.ilsleaf = new AENodeImageLabelSet(fMapper, batchsize, nodes[nodes.length - 2], nodes[nodes.length - 1]);

		// 乱数ジェネレータの作成
		generator = AutoEncoderNode.createGenerator();
		
		// 最終段の逆誤差散乱用
		int[] nodesleaf = {nodes[nodes.length - 2], nodes[nodes.length - 1]};
		nnleaf = new NeuralNet(fMapper, lRate, nodesleaf, batchsize);
		this.regionleaf = nnleaf.createDefaultCUDARegion(); 

		// 結果のニューラルネット
		nn = new NeuralNet(fMapper, lRate, nodes, batchsize);
		region = nn.createDefaultCUDARegion();
		
		// AutoEncoder
		aenodes = new AutoEncoderNode[nodes.length - 2];
		IntStream.range(0, nodes.length - 2).forEach(i->{
			try {
				aenodes[i] = new AutoEncoderNode(fMapper, generator, lRate, batchsize, nodes[i], nodes[i + 1]);
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		});
		
		// 入力域の編集
		devInArray = new CUdeviceptr[batchsize];
		devIn = new CUdeviceptr();
		cuMemAlloc(devIn, Sizeof.POINTER * batchsize);
	}
	
	public void finalize() {
		cuMemFree(devIn);
	}
	
	/**
	 * create sample sequence
	 * 
	 * @param q
	 * @return
	 */
	private int[] createSamples(int q) {
		int[] samples = new int[batchsize];
		IntStream.range(0, batchsize).forEach(i->{
			samples[i] = rand.nextInt(q);
		});
		return samples;
	}
	
	/**
	 * AutoEncoder training
	 * @param ils
	 */
	public void training(ImageLabelSet ils) {
		int q = ils.getQuantity();
		int[] totologyleaf = IntStream.range(0, nn.neurons[nn.neurons.length - 1].getInn())
				.toArray();
		
		// phase1: AutoEncoder を使って係数を荒く推定する
		IntStream.range(0, nsamples1).forEach(i->{
			// 入力値の準備
			int[] samples = createSamples(q);
			IntStream.range(0, batchsize).forEach(j->{
				devInArray[i] = ils.image.getContentDev(samples[j]);
			});
			cuMemcpyHtoD(devIn, Pointer.to(devInArray), Sizeof.POINTER * batchsize);
			
			// 上から順番に AutoEncoderNode をかけていく
			IntStream.range(0,  nodes.length - 2).forEachOrdered(j->{
				// デバイスメモリの準備
				CUdeviceptr di = (j == 0)?devIn:aenodes[j - 1].getOut0();
				aenodes[j].setContentDev(di, threshold);
				
				// 逆誤差散乱法による AutoEncoder 係数の算出
				aenodes[j].backPropagate(nset);
			});
			
			// 最後は通常の逆誤差散乱法を実行する
			ilsleaf.setContentDev(aenodes[nodes.length - 2].getOut0(), ils.label.getContentDev());
			regionleaf.z[0] = aenodes[nodes.length - 2].getOut0();
			IntStream.range(0, nset).forEach(j->{
				nnleaf.backPropagate(regionleaf, ilsleaf, totologyleaf);
			});
			
			// 進行状況を表示
			System.out.print(".");
		});
		
		// phase1.5: 算出した係数を nn に集める
		IntStream.range(0,  nodes.length - 2).forEach(i->{
			aenodes[i].copyCoef(nn.neurons[i]);
		});
		nnleaf.neurons[nnleaf.neurons.length - 1].copyCoef(nn.neurons[nn.neurons.length - 1]);
		
		// phase2: ファインチューニングを行う
		IntStream.range(0, nsamples2).forEach(i->{
			nn.backPropagate(region, ils, nset);
			
			// 進行状況を表示
			System.out.print("@");
		});
		
		// training 終了
		System.out.println();
	}
	
	/**
	 * return argmax function
	 * 
	 * @param values
	 * @return
	 */
	private int argmax(float[] values) {
		float xmax = -Float.MAX_VALUE;
		int imax = 0;
		for (int i = 0; i < values.length; ++i) {
			if (xmax < values[i]) {
				imax = i;
				xmax = values[i];
			}
		}
		return imax;
	}
	
	/**
	 * repeat tests 'count' counts
	 * @param ils
	 * @param count
	 */
	public void test(ImageLabelSet ils, int count) {
		int outn = ils.label.getOutputCount();
		CUdeviceptr sm = new CUdeviceptr();
		cuMemAlloc(sm, Sizeof.POINTER * batchsize);
		float[][] result = new float[batchsize][outn];
		
		IntStream.range(0, count).forEachOrdered(c->{
			// samples を準備する
			int[] samples = createSamples(ils.getQuantity());
			CUdeviceptr[] sms = new CUdeviceptr[batchsize];
			IntStream.range(0, batchsize).forEach(j->{
				sms[j] = ils.image.getContentDev(samples[j]);
			});
			cuMemcpyHtoD(sm, Pointer.to(sms), Sizeof.POINTER * batchsize);
			
			// forward 操作
			region.z[0] = sm;
			nn.forward(region);
			SimpleNet net = nn.neurons[nn.neurons.length - 1];
			IntStream.range(0, batchsize).forEach(s->{
				cuMemcpyDtoH(Pointer.to(result[s]), net.devOutz2D[s], Sizeof.FLOAT * outn);
			});
			double cc = (double)IntStream.range(0,  batchsize)
					.filter(s->argmax(result[s]) == argmax(ils.label.getContent(s)))
					.count();
			cc = cc / (double)batchsize * 100.0;
			System.out.printf("loop:%d->%.3f%%\n", c, cc);
		});
		
		cuMemFree(sm);
	}
}
