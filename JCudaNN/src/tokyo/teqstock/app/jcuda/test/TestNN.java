package tokyo.teqstock.app.jcuda.test;

import java.io.IOException;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import tokyo.teqstock.app.jcuda.lib.MNIST;
import tokyo.teqstock.app.jcuda.lib.NNUtil;
import tokyo.teqstock.app.jcuda.lib.NeuralNet;
import tokyo.teqstock.app.jcuda.lib.SimpleNet;

import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;
import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;

/**
 * NeuralNetwork クラスのテスト
 * @author minoru
 *
 */
public class TestNN {
	private static final String CUFILENAME = "JCudaNNKernel.cu";
	private static final int NSET = 64;
	private static final int BATCHSIZE = 128;
	private static final int NSAMPLE = 64;
	private static final int COUNT = 10;
	private static final float LRATE = 0.1f;
	private static final int[] NODES = { 784, 100, 10 };

	MNIST teacher, apply;
	Map<String, CUfunction> fMapper;
	NeuralNet nn;
	NeuralNet.CUDARegion region;
	CUdeviceptr batchPnt;
	
	public TestNN() {
	}

	public int argmax(float[] y) {
		float xmax = -Float.MAX_VALUE;
		int imax = 0;
		
		for (int i = 0; i < y.length; ++i) {
			if (xmax < y[i]) {
				imax = i;
				xmax = y[i];
			}
		}
		return imax;
	}
	
	/**
	 * テストの準備
	 * @throws IOException
	 */
	public void prepare() throws IOException {
		// JCuda の初期化
		fMapper = NNUtil.initJCuda(CUFILENAME);
		
		// ニューラルネットの初期化
		nn = new NeuralNet(fMapper, LRATE, NODES, BATCHSIZE);
		
		// 教師データ、テストデータの読み込み
		teacher = MNIST.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", true, true);
		apply = MNIST.load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", true, true);

		// デフォルト CUDARegion の取得
		region = nn.createDefaultCUDARegion();
	
		// batchPnt の初期化
		batchPnt = new CUdeviceptr();
		cuMemAlloc(batchPnt, Sizeof.POINTER * BATCHSIZE);
		
		// DEBUG: 初期重みでの出力
		int[] samples = createSamples(teacher.getQuantity(), BATCHSIZE);
		checkOne(teacher, samples);
	}
	
	public void release() {
		cuMemFree(batchPnt);
	}
	
	/**
	 * テスト本体
	 */
	public void run() {
		// トレーニング
		NeuralNet.CUDARegion region = nn.createDefaultCUDARegion();
		IntStream.range(0, NSAMPLE).forEachOrdered(s->{
			int[] samples = nn.backPropagate(region, teacher, NSET);
//			double rate = checkOne(teacher, samples);
//			System.out.printf("rate=%.4f\n", rate);
			System.out.print(".");
		});
	}

	private int[] createSamples(int q, int size) {
		Random rand = new Random();
		int[] array = new int[size];
		IntStream.range(0, size).forEach(i->{
			array[i] = rand.nextInt(q);
		});
		return array;
	}

	/**
	 * １回のループで調査する正答率
	 * @return
	 */
	private double checkOne(MNIST mnist, int[] samples) {
		// 入力データ
		CUdeviceptr[] batchIn = new CUdeviceptr[BATCHSIZE];
		
		// インデックスからポインタに変換する
		IntStream.range(0, BATCHSIZE).forEach(i->{
			batchIn[i] = mnist.image.getContentDev(samples[i]);
		});
		cuMemcpyHtoD(region.z0, Pointer.to(batchIn), Sizeof.POINTER * BATCHSIZE);
		
		// forward 操作
		region.z[0] = region.z0;
		nn.forward(region);

		// デバイスメモリの内容をホストメモリにコピーする
		SimpleNet net = nn.neurons[nn.neurons.length - 1];
		double count = (double)IntStream.range(0, BATCHSIZE)
				.filter(k->{
					int outn = net.getOutn();
					float[] r = new float[outn];
					float[] h = new float[outn];
					CUdeviceptr hot = mnist.label.getContentDev(samples[k]);
					cuMemcpyDtoH(Pointer.to(r), net.devOutz2D[k], Sizeof.FLOAT * outn);
					cuMemcpyDtoH(Pointer.to(h), hot, Sizeof.FLOAT * outn);
					return argmax(r) == argmax(h);
				})
				.count();
			
		
		// 全体の数で割り、100 を掛けて正答率を求める
		return count / (double)BATCHSIZE * 100.0;
	}
	
	/**
	 * 新しいデータセットへの適用性を調べる
	 */
	public void check() {
		// データセットを apply から用意する
		System.out.println();
		IntStream.range(0,  COUNT).forEach(c->{
			// ランダムサンプリング
			int[] samples = createSamples(apply.getQuantity(), BATCHSIZE);
			double rate = checkOne(apply, samples);
			System.out.printf("loop %d: rate=%.2f%%\n", c, rate);
		});
		
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		TestNN test = new TestNN();
		test.prepare();
		test.run();
		test.check();
		test.release();
	}
}
