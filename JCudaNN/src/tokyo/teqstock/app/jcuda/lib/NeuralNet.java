package tokyo.teqstock.app.jcuda.lib;

import java.io.IOException;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuLaunchKernel;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import tokyo.teqstock.app.jcuda.lib.SimpleNet.OutputFormat;

/**
 * Neural Network implementations
 * @author minoru
 *
 */
public class NeuralNet {
	private static final int NTHREAD = 32;
	private static final int NTHREAD2 = 8;
	
	private int calcBlock(int x) {
		return (x + NTHREAD - 1) / NTHREAD;
	}
	
	private int calcBlock2D(int x) {
		return (x + NTHREAD2 - 1) / NTHREAD2;
	}
	
	/**
	 * learning rate
	 */
	private  final float LRATE;

	private final int[] nodes;
	
	/**
	 * kernel function mappings
	 */
	private final Map<String, CUfunction> fMapper;
	
	/**
	 * layered neurons
	 */
	public SimpleNet[] neurons;

	public class CUDARegion {
		private final int[] nodes;
		
		public CUdeviceptr[] z;
		
		/**
		 * compensation for the back propagation method (weight)
		 */
		public CUdeviceptr[][] devWDeltaArray2D, devWDerivArray2D;
		public CUdeviceptr[] devWDeltaArray, devWDerivArray;
		
		/**
		 * compensation for the back propagation method (bias)
		 */
		public CUdeviceptr[] devBDeltaArray, devBDerivArray;

		/**
		 * constructor
		 * 
		 * @param nodes
		 * @param sn
		 * @param in
		 */
		public CUDARegion(int[] nodes, SimpleNet[] sn, CUdeviceptr in) {
			this.nodes = nodes;
			z = new CUdeviceptr[nodes.length];
			z[0] = in;
			
			// for delta w
			devWDeltaArray2D = new CUdeviceptr[nodes.length - 1][];
			IntStream.range(0,  nodes.length - 1).forEach(k->{
				devWDeltaArray2D[k] = new CUdeviceptr[nodes[k]];
				IntStream.range(0, nodes[k]).forEach(i->{
					devWDeltaArray2D[k][i] = new CUdeviceptr();
					cuMemAlloc(devWDeltaArray2D[k][i], Sizeof.FLOAT * nodes[k + 1]);
				});
				devWDeltaArray = new CUdeviceptr[nodes[k]];
				cuMemAlloc(devWDeltaArray[k], Sizeof.POINTER * nodes[k]);
				cuMemcpyHtoD(devWDeltaArray[k], Pointer.to(devWDeltaArray2D[k]), Sizeof.POINTER * nodes[k]);
			});

			// for deriv w
			devWDerivArray2D = new CUdeviceptr[nodes.length - 1][];
			IntStream.range(0,  nodes.length - 1).forEach(k->{
				devWDerivArray2D[k] = new CUdeviceptr[nodes[k]];
				IntStream.range(0, nodes[k]).forEach(i->{
					devWDerivArray2D[k][i] = new CUdeviceptr();
					cuMemAlloc(devWDerivArray2D[k][i], Sizeof.FLOAT * nodes[k + 1]);
				});
				devWDerivArray = new CUdeviceptr[nodes[k]];
				cuMemAlloc(devWDerivArray[k], Sizeof.POINTER * nodes[k]);
				cuMemcpyHtoD(devWDerivArray[k], Pointer.to(devWDeltaArray2D[k]), Sizeof.POINTER * nodes[k]);
			});
			
			// for delta b
			devBDeltaArray = new CUdeviceptr[nodes.length - 1];
			IntStream.range(0, nodes.length - 1).forEach(j->{
				devBDeltaArray[j] = new CUdeviceptr();
				cuMemAlloc(devBDeltaArray[j], Sizeof.FLOAT * nodes[j + 1]);
			});

			// for deriv b
			devBDerivArray = new CUdeviceptr[nodes.length - 1];
			IntStream.range(0, nodes.length - 1).forEach(j->{
				devBDerivArray[j] = new CUdeviceptr();
				cuMemAlloc(devBDerivArray[j], Sizeof.FLOAT * nodes[j + 1]);
			});
		}
		
		public void finalize() {
			IntStream.range(0, nodes.length - 1).forEach(k->{
				IntStream.range(0,  nodes[k]).forEach(i->{
					// free delta w
					cuMemFree(devWDeltaArray2D[k][i]);
					
					// free deriv w
					cuMemFree(devWDerivArray2D[k][i]);
				});
				cuMemFree(devWDeltaArray[k]);
				cuMemFree(devWDerivArray[k]);
				
				// free delta b
				cuMemFree(devBDeltaArray[k]);
				
				// free deriv b
				cuMemFree(devBDerivArray[k]);
			});
		}
	}
	
	/**
	 * random number generator
	 */
	private Random rand;
	
	/**
	 * constructor
	 * @param nodes	# of nodes in array format
	 * @throws IOException
	 */
	public NeuralNet(Map<String, CUfunction> fMapper, float lrate, int[] nodes) throws IOException {
		this.LRATE = lrate;
		this.nodes = nodes;
		this.fMapper = fMapper;
		this.rand = new Random();
		this.neurons = new SimpleNet[nodes.length - 1];
		IntStream.range(0,  neurons.length).forEach(k->{
			neurons[k] = new SimpleNet(fMapper, nodes[k], nodes[k + 1]);
			if (k == nodes.length - 2) {
				neurons[k].format = OutputFormat.SOFTMAX;
			}
		});
	}

	/**
	 * neural network forward operation
	 * @param in
	 * @return
	 */
	public CUdeviceptr forward(CUDARegion region) {
		IntStream.range(0, neurons.length).forEachOrdered(k->{
			region.z[k + 1] = neurons[k].forward(region.z[k]);
		});
		return region.z[neurons.length];
	}

	/**
	 * host interface for the forward operation
	 * @param in
	 * @return
	 */
	public float[] forward(float[] in) {
		// デバイスメモリへの転送
		CUdeviceptr devIn = new CUdeviceptr();
		cuMemAlloc(devIn, Sizeof.FLOAT * in.length);
		cuMemcpyHtoD(devIn, Pointer.to(in), Sizeof.FLOAT * in.length);
		
		// forward 操作
		CUDARegion region = new CUDARegion(nodes, neurons, devIn);
		CUdeviceptr outz = forward(region);
		
		// ホストメモリへ転送
		int outn = nodes[nodes.length - 1];
		float[] out = new float[outn];
		cuMemcpyDtoH(Pointer.to(out), outz, Sizeof.FLOAT * outn);
		
		// デバイスメモリを解放
		cuMemFree(devIn);
		return out;
	}
	
	/**
	 * １次元メモリのクリア
	 * @param p
	 * @param size
	 */
	private void clearMem1D(Pointer p, int size) {
		Pointer kp = Pointer.to(p, Pointer.to(new int[]{size}));
		cuLaunchKernel(fMapper.get("clear1D"),
				calcBlock(size), 1, 1,
				NTHREAD, 1, 1,
				0, null,
				kp, null);
	}
	
	/**
	 * ２次元メモリのクリア
	 * @param p
	 * @param xsize
	 * @param ysize
	 */
	private void clearMem2D(Pointer p, int xsize, int ysize) {
		Pointer kp = Pointer.to(p, Pointer.to(new int[]{xsize}), Pointer.to(new int[]{ysize}));
		cuLaunchKernel(fMapper.get("clear1D"),
				calcBlock2D(xsize), calcBlock2D(ysize), 1,
				NTHREAD2, NTHREAD2, 1,
				0, null,
				kp, null);
	}
	
	/**
	 * assuming forward() already called and the calculated values are cached
	 * @param ilst
	 * @param index
	 */
	public void backPropagate1(CUDARegion region, ImageLabelSet ils, int index) {
		// Phase1: calculate the derivatives
		// 最外殻から計算を始める 
		int m = neurons.length - 1;
		CUdeviceptr teacher = ils.label.getContentDev(index);
		CUdeviceptr delta = region.devBDeltaArray[m];
		clearMem1D(delta, neurons[m].getOutn());
		clearMem2D(region.devWDeltaArray[m], neurons[m].getInn(), neurons[m].getOutn());
		neurons[m].calc_deriv_b(delta, teacher, true);
		neurons[m].calc_deriv_w(region.devWDeltaArray[m], region.z[m], nodes[m], delta, nodes[m + 1]);
		
		while (--m >= 0) {
			// 隠れ層の計算
			delta = region.devBDeltaArray[m];
			clearMem1D(delta, neurons[m].getOutn());
			clearMem2D(region.devWDeltaArray[m], neurons[m].getInn(), neurons[m].getOutn());
			neurons[m].calc_deriv_b(delta, region.z[m], false);;
			neurons[m].calc_deriv_w(region.devWDeltaArray[m], region.z[m], nodes[m], delta, nodes[m + 1]);
		}
		
		// Phase2: accumlation
		IntStream.range(0, neurons.length).forEach(k->{
			int outn = neurons[k].getOutn();
			int inn = neurons[k].getInn();
			Pointer kp = Pointer.to(
					Pointer.to(region.devBDerivArray[k]),
					Pointer.to(region.devBDeltaArray[k]),
					Pointer.to(new int[]{outn})
					);
			cuLaunchKernel(fMapper.get("vec_add_1d"),
					calcBlock(outn), 1, 1,
					NTHREAD, 1, 1,
					0, null,
					kp, null);
			kp = Pointer.to(
					Pointer.to(region.devWDerivArray[k]),
					Pointer.to(region.devWDeltaArray[k]),
					Pointer.to(new int[]{inn}), Pointer.to(new int[]{outn}));
			cuLaunchKernel(fMapper.get("vec_add_2d"),
					calcBlock2D(inn), calcBlock2D(outn), 1,
					NTHREAD2, NTHREAD2, 1,
					0, null,
					kp, null);
		});
	}

	/**
	 * back propagation for batches
	 * @param ils
	 * @param samples
	 */
	public void backPropagate(CUDARegion region, ImageLabelSet ils, int[] samples) {
		// initialize deltas
		IntStream.range(0, neurons.length).forEach(k->{
			clearMem1D(region.devBDerivArray[k], neurons[k].getOutn());
			clearMem2D(region.devBDerivArray[k], neurons[k].getInn(), neurons[k].getOutn());
		});
		
		// phase1: accumlate deltas
		IntStream.range(0, samples.length).forEach(m->{
			// copy input data
			region.z[0] = ils.image.getContentDev(samples[m]);
			
			// phase1: forward propagation
			forward(region);
			
			// phase2: back propagation
			backPropagate1(region, ils, samples[m]);
		});

		// phase2: learning process
		IntStream.range(0, neurons.length).forEach(k->{
			SimpleNet net = neurons[k];
			int inn = net.getInn();
			int outn = net.getOutn();
			
			// w の学習
			Pointer kp = Pointer.to(
					Pointer.to(neurons[k].devW),
					Pointer.to(region.devWDerivArray),
					Pointer.to(new float[]{LRATE}),
					Pointer.to(new int[]{inn}),
					Pointer.to(new int[]{outn}),
					Pointer.to(new float[]{samples.length})
					);
			cuLaunchKernel(fMapper.get("learn_2d"),
					calcBlock2D(inn), calcBlock2D(outn), 1,
					NTHREAD2, NTHREAD2, 1,
					0, null,
					kp, null
					);
			
			// b の学習
			kp = Pointer.to(
					Pointer.to(neurons[k].devB),
					Pointer.to(region.devBDerivArray),
					Pointer.to(new float[]{LRATE}),
					Pointer.to(new int[]{outn}),
					Pointer.to(new float[]{samples.length})
					);
			cuLaunchKernel(fMapper.get("learn_1d"),
					calcBlock(outn), 1, 1,
					NTHREAD, 1, 1,
					0, null,
					kp, null);
		});
	}

	/**
	 * back propagation calculation for the batch size and the number of iterations
	 * @param mnist
	 * @param nset
	 * @param batchsize
	 */
	public int[] backPropagate(ImageLabelSet ils, int nset, int batchsize) {
		int[] samples = new int[batchsize];
		IntStream.range(0,  batchsize).forEach(i->{
			samples[i] = rand.nextInt(ils.image.getQuantity());
		});
		CUDARegion region = new CUDARegion(nodes, neurons, null);
		IntStream.range(0, nset).forEach(i->{
			backPropagate(region, ils, samples);
			if (i % 10 == 0) {
					System.out.print(".");
			}
		});
		return samples;
	}
}
