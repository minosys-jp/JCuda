package tokyo.teqstock.app.jcuda.test;

import java.io.IOException;
import java.util.Map;
import java.util.Random;
import java.util.stream.IntStream;

import tokyo.teqstock.app.jcuda.lib.MNIST;
import tokyo.teqstock.app.jcuda.lib.NNUtil;
import tokyo.teqstock.app.jcuda.lib.NeuralNet;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

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
	private static final String CUFILENAME = "jCudaNNKernel.cu";
	private static final int NSET = 20;
	private static final int BATCHSIZE = 100;
	private static final int NSAMPLE = 10;
	private static final int COUNT = 10;
	private static final float LRATE = 0.1f;
	private static final int[] NODES = { 768, 100, 10 };

	MNIST teacher, apply;
	Map<String, CUfunction> fMapper;
	NeuralNet nn;
	
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
		teacher = MNIST.load("train-images-idx3-ubyte", "train-labels-idx1-ubyte", true, true);
		apply = MNIST.load("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", true, true);
		fMapper = NNUtil.initJCuda(CUFILENAME);
		nn = new NeuralNet(fMapper, LRATE, NODES);
	}
	
	/**
	 * テスト本体
	 */
	public void run() {
		// トレーニング
		IntStream.range(0, NSAMPLE).forEachOrdered(s->{
			nn.backPropagate(teacher, NSET, BATCHSIZE);
		});
	}

	private int[] createSamples(final int q, final int size) {
		Random rand = new Random();
		int[] array = new int[size];
		IntStream.range(0, size).forEach(i->{
			array[i] = rand.nextInt(q);
		});
		return array;
	}

	/**
	 * サンプルから forward 実行して出力値を推定する
	 * @param sample
	 * @return
	 */
	private int getArgMax(CUdeviceptr sample) {
		NeuralNet.CUDARegion region = new NeuralNet.CUDARegion(NODES, nn.neurons, sample);
		CUdeviceptr devZ = nn.forward(region);
		float[] z = new float[NODES[NODES.length - 1]];
		cuMemcpyDtoH(Pointer.to(z), devZ, Sizeof.FLOAT * z.length);
		return argmax(z);
	}
	
	/**
	 * １回のループで調査する正答率
	 * @return
	 */
	private double checkOne() {
		// 入力データ
		CUdeviceptr[] batchIn = new CUdeviceptr[BATCHSIZE];
		
		// 正解
		CUdeviceptr[] answer = new CUdeviceptr[BATCHSIZE];
		
		// ランダムサンプリング
		int[] samples = createSamples(apply.getQuantity(), BATCHSIZE);
		
		// インデックスからポインタに変換する
		IntStream.range(0, BATCHSIZE).forEach(i->{
			batchIn[i] = apply.image.getContentDev(samples[i]);
			answer[i] = apply.label.getContentDev(samples[i]);
		});
		
		// デバイスメモリの内容をホストメモリにコピーする
		float[][] labels = new float[BATCHSIZE][];
		int outn = NODES[NODES.length - 1];
		IntStream.range(0, BATCHSIZE).forEach(k->{
			labels[k] = new float[outn];
			cuMemcpyDtoH(Pointer.to(labels[k]), answer[k], Sizeof.FLOAT * outn);
		});
		
		// 正答数を集計
		double count = (double)IntStream.range(0, BATCHSIZE)
				.filter(k->getArgMax(batchIn[k]) == argmax(labels[k]))
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
			double rate = checkOne();
			System.out.printf("loop %d: rate=%.2f%%\n", c, rate);
		});
		
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		TestNN test = new TestNN();
		test.prepare();
		test.run();
		test.check();
	}

}
