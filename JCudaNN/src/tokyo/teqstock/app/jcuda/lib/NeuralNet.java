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
import static jcuda.driver.JCudaDriver.cuCtxSynchronize;

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
	
	private final int batchsize;
	
	/**
	 * kernel function mappings
	 */
	private final Map<String, CUfunction> fMapper;
	
	/**
	 * layered neurons
	 */
	public SimpleNet[] neurons;

	public static class CUDARegion {
		private final int[] nodes;
		
		public final int batchsize;
		
		public CUdeviceptr teacher;

		public CUdeviceptr z0;
		
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
		 * for bactch size
		 */
		public CUdeviceptr[][] devBDeltaArray2D, devBDerivArray2D;

		public CUdeviceptr[] sumPtr;
		
		/**
		 * constructor
		 * 
		 * @param nodes
		 * @param sn
		 * @param in
		 */
		public CUDARegion(int[] nodes, SimpleNet[] sn, int batchsize) {
			this.nodes = nodes;
			this.batchsize = batchsize;
			teacher = new CUdeviceptr();
			cuMemAlloc(teacher, Sizeof.INT * batchsize);
		
			z0 = new CUdeviceptr();
			cuMemAlloc(z0, Sizeof.POINTER * batchsize);
			
			z = new CUdeviceptr[nodes.length];
			z[0] = z0;
			
			// for delta w
			devWDeltaArray2D = new CUdeviceptr[nodes.length - 1][];
			devWDeltaArray = new CUdeviceptr[nodes.length - 1];
			IntStream.range(0,  nodes.length - 1).forEach(k->{
				devWDeltaArray2D[k] = new CUdeviceptr[nodes[k]];
				IntStream.range(0, nodes[k]).forEach(i->{
					devWDeltaArray2D[k][i] = new CUdeviceptr();
					cuMemAlloc(devWDeltaArray2D[k][i], Sizeof.FLOAT * nodes[k + 1]);
				});
				devWDeltaArray[k] = new CUdeviceptr();
				cuMemAlloc(devWDeltaArray[k], Sizeof.POINTER * nodes[k]);
				cuMemcpyHtoD(devWDeltaArray[k], Pointer.to(devWDeltaArray2D[k]), Sizeof.POINTER * nodes[k]);
			});

			// for deriv w
			devWDerivArray2D = new CUdeviceptr[nodes.length - 1][];
			devWDerivArray = new CUdeviceptr[nodes.length - 1];
			IntStream.range(0,  nodes.length - 1).forEach(k->{
				devWDerivArray2D[k] = new CUdeviceptr[nodes[k]];
				IntStream.range(0, nodes[k]).forEach(i->{
					devWDerivArray2D[k][i] = new CUdeviceptr();
					cuMemAlloc(devWDerivArray2D[k][i], Sizeof.FLOAT * nodes[k + 1]);
				});
				devWDerivArray[k] = new CUdeviceptr();
				cuMemAlloc(devWDerivArray[k], Sizeof.POINTER * nodes[k]);
				cuMemcpyHtoD(devWDerivArray[k], Pointer.to(devWDerivArray2D[k]), Sizeof.POINTER * nodes[k]);
			});
			
			// for delta b
			devBDeltaArray = new CUdeviceptr[nodes.length - 1];
			devBDeltaArray2D = new CUdeviceptr[nodes.length - 1][];
			IntStream.range(0, nodes.length - 1).forEach(j->{
				devBDeltaArray2D[j] = new CUdeviceptr[batchsize];
				IntStream.range(0, batchsize).forEach(s->{
					devBDeltaArray2D[j][s] = new CUdeviceptr();
					cuMemAlloc(devBDeltaArray2D[j][s], Sizeof.FLOAT * nodes[j + 1]);
				});
				devBDeltaArray[j] = new CUdeviceptr();
				cuMemAlloc(devBDeltaArray[j], Sizeof.POINTER * batchsize);
				cuMemcpyHtoD(devBDeltaArray[j], Pointer.to(devBDeltaArray2D[j]), Sizeof.POINTER * batchsize);
			});

			// for deriv b
			devBDerivArray = new CUdeviceptr[nodes.length - 1];
			devBDerivArray2D = new CUdeviceptr[nodes.length - 1][];
			IntStream.range(0, nodes.length - 1).forEach(j->{
				devBDerivArray2D[j] = new CUdeviceptr[batchsize];
				IntStream.range(0, batchsize).forEach(s->{
					devBDerivArray2D[j][s] = new CUdeviceptr();
					cuMemAlloc(devBDerivArray2D[j][s], Sizeof.FLOAT * nodes[j + 1]);
				});
				devBDerivArray[j] = new CUdeviceptr();
				cuMemAlloc(devBDerivArray[j], Sizeof.POINTER * batchsize);
				cuMemcpyHtoD(devBDerivArray[j], Pointer.to(devBDerivArray2D[j]), Sizeof.POINTER * batchsize);
			});
			
			// for test
			sumPtr = new CUdeviceptr[nodes.length - 1];
			IntStream.range(0, nodes.length - 1).forEach(k->{
				sumPtr[k] = new CUdeviceptr();
				cuMemAlloc(sumPtr[k], Sizeof.FLOAT * nodes[k + 1]);
			});
		}
		
		public void finalize() {
			IntStream.range(0, nodes.length - 1).forEach(k->{
				cuMemFree(sumPtr[k]);
			});
			
			// free 0th order output (=input)
			cuMemFree(z0);
			
			// free teacher indices
			cuMemFree(teacher);
			
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
				IntStream.range(0, batchsize).forEach(i->{
					cuMemFree(devBDeltaArray2D[k][i]);
					cuMemFree(devBDerivArray2D[k][i]);
				});
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
	 * 
	 * @param fMapper
	 * @param lrate
	 * @param nodes
	 * @throws IOException
	 */
	public NeuralNet(Map<String, CUfunction> fMapper, float lrate, int[] nodes, int batchsize) throws IOException {
		this.LRATE = lrate;
		this.nodes = nodes;
		this.batchsize = batchsize;
		this.fMapper = fMapper;
		this.rand = new Random();
		this.neurons = new SimpleNet[nodes.length - 1];
		IntStream.range(0,  neurons.length).forEach(k->{
			neurons[k] = new SimpleNet(fMapper, nodes[k], nodes[k + 1], batchsize);
			if (k == nodes.length - 2) {
				neurons[k].format = OutputFormat.SOFTMAX;
			}
		});
	}

	/**
	 * neural network forward operation
	 * @param region
	 * @return
	 */
	public CUdeviceptr forward(CUDARegion region) {
		IntStream.range(0, neurons.length).forEachOrdered(k->{
			region.z[k + 1] = neurons[k].forward(region.z[k]);
		});
		return region.z[neurons.length];
	}

	/**
	 * ２次元メモリのクリア
	 * @param p
	 * @param xsize
	 * @param ysize
	 */
	public void clearMem2D(Pointer p, int xsize, int ysize) {
		Pointer kp = Pointer.to(Pointer.to(p), Pointer.to(new int[]{xsize}), Pointer.to(new int[]{ysize}));
		cuLaunchKernel(fMapper.get("clear2D"),
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
	public void backPropagate1(CUDARegion region, ImageLabelSet ils, int[] index) {
		// Phase1: calculate the derivatives
		// 最外殻から計算を始める 
		int m = neurons.length - 1;
		cuMemcpyHtoD(region.teacher, Pointer.to(index), Sizeof.INT * region.batchsize);
		CUdeviceptr delta = region.devBDeltaArray[m];
		neurons[m].calc_deriv_b(delta, ils.label.getContentDev(), region.teacher, true);
		neurons[m].calc_deriv_w(region.devWDeltaArray[m], region.z[m],
				neurons[m].getInn(), delta, neurons[m].getOutn());
		
		while (--m >= 0) {
			// 隠れ層の計算
			delta = region.devBDeltaArray[m];
			neurons[m + 1].calc_deriv_b(delta, region.z[m + 1], region.devBDeltaArray[m + 1], false);
			neurons[m].calc_deriv_w(region.devWDeltaArray[m], region.z[m],
					neurons[m].getInn(), delta, neurons[m].getOutn());
		}
		
		// Phase2: accumulation
		IntStream.range(0, neurons.length).forEach(k->{
			int inn = neurons[k].getInn();
			int outn = neurons[k].getOutn();
			Pointer kp = Pointer.to(
					Pointer.to(region.devBDerivArray[k]),
					Pointer.to(region.devBDeltaArray[k]),
					Pointer.to(new int[]{batchsize}),
					Pointer.to(new int[]{outn})
					);
			cuLaunchKernel(fMapper.get("vec_add_2d"),
					calcBlock2D(batchsize), calcBlock2D(outn), 1,
					NTHREAD2, NTHREAD2, 1,
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
			cuCtxSynchronize();
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
			clearMem2D(region.devBDerivArray[k], batchsize, neurons[k].getOutn());
			clearMem2D(region.devWDerivArray[k], neurons[k].getInn(), neurons[k].getOutn());
		});
		cuCtxSynchronize();
		
		// phase1: accumulate deltas
		CUdeviceptr[] z0s = new CUdeviceptr[samples.length];
		IntStream.range(0, samples.length).forEach(i->{
			z0s[i] = ils.image.getContentDev(samples[i]);
		});

		// copy input data
		cuMemcpyHtoD(region.z0, Pointer.to(z0s), Sizeof.POINTER * samples.length);
		region.z[0] = region.z0;
		
		// forward operation
		forward(region);
		
		// back propagation
		backPropagate1(region, ils, samples);

		// phase2: learning process
		IntStream.range(0, neurons.length).forEach(k->{
			SimpleNet net = neurons[k];
			int inn = net.getInn();
			int outn = net.getOutn();
		
			// w の学習
			Pointer kp = Pointer.to(
					Pointer.to(neurons[k].devW),
					Pointer.to(region.devWDerivArray[k]),
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
					Pointer.to(region.devBDerivArray[k]),
					Pointer.to(new float[]{LRATE}),
					Pointer.to(new int[]{outn}),
					Pointer.to(new float[]{samples.length}),
					Pointer.to(new int[]{batchsize})
					);
			cuLaunchKernel(fMapper.get("learn_1d"),
					calcBlock(outn), 1, 1,
					NTHREAD, 1, 1,
					0, null,
					kp, null);
			cuCtxSynchronize();
		});
	}

	/**
	 * back propagation calculation for the batch size and the number of iterations
	 * @param mnist
	 * @param nset
	 * @param batchsize
	 */
	public int[] backPropagate(CUDARegion region, ImageLabelSet ils, int nset) {
		int[] samples = new int[batchsize];
		IntStream.range(0,  batchsize).forEach(i->{
			samples[i] = rand.nextInt(ils.image.getQuantity());
		});
		IntStream.range(0, nset).forEach(i->{
			backPropagate(region, ils, samples);
		});
		return samples;
	}
	
	public float[] sumof_test(CUDARegion region) {
		float[] bb = new float[nodes.length - 1];
		IntStream.range(0, nodes.length - 1).forEach(n->{
			SimpleNet net = neurons[n];
			Pointer kp = Pointer.to(
					Pointer.to(region.sumPtr[n]),
					Pointer.to(region.devWDerivArray[n]),
					Pointer.to(region.devBDerivArray[n]),
					Pointer.to(new int[]{net.getInn()}),
					Pointer.to(new int[]{net.getOutn()}),
					Pointer.to(new int[]{batchsize})
					);
			cuLaunchKernel(fMapper.get("test_sum"),
					calcBlock(net.getOutn()), 1, 1,
					NTHREAD, 1, 1,
					0, null,
					kp, null
					);
			cuCtxSynchronize();
			float[] bo = new float[net.getOutn()];
			cuMemcpyDtoH(Pointer.to(bo), region.sumPtr[n], Sizeof.FLOAT * net.getOutn());

			// sum for outn
			float r = (float)IntStream.range(0, net.getOutn()).mapToDouble(j->bo[j]).sum();
			bb[n] = r;
		});
		return bb;
	}
	
	/**
	 * デフォルトの CUDARegion を作成する
	 * @return
	 */
	public CUDARegion createDefaultCUDARegion() {
		return new CUDARegion(nodes, neurons, batchsize);
	}
}
