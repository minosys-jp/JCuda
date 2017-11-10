package tokyo.teqstock.app.jcuda.lib;

import java.io.BufferedInputStream;
import java.io.FileInputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.OptionalDouble;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoH;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;

/**
 * MNIST file objects
 * @author minoru
 *
 */
public class MNIST extends ImageLabelSet {
	private final static String train_images = "train-images-idx3-ubyte";
	private final static String train_labels = "train-labels-idx1-ubyte";
	private final static String t10k_images = "t10k-images-idx3-ubyte";
	private final static String t10k_labels = "t10k-labels-idx1-ubyte";

	/**
	 * load MNIST image file
	 * @author minoru
	 *
	 */
	public class MNISTImage extends ImageLabelSet.BaseImage {
		private  boolean normalize;
		private  int w, h, n;
		public byte b[][];
		float f[][];
		CUdeviceptr[] devMemArray;
		CUdeviceptr devMem;
		
		MNISTImage(boolean normalize) {
			this.normalize = normalize;
		}

		public void finalize() {
			if (devMem != null) {
				IntStream.range(0,  devMemArray.length).forEach(i->{
					cuMemFree(devMemArray[i]);
				});
				cuMemFree(devMem);
				devMem = null;
			}
		}
		
		/**
		 * image file loader
		 * @param bis
		 * @throws IOException
		 */
		public void loader(BufferedInputStream bis) throws IOException {
			// magic number を無視
			MNIST.readInt(bis);;
			// 数
			n = MNIST.readInt(bis);
			// 幅
			w = MNIST.readInt(bis);
			// 高さ
			h = MNIST.readInt(bis);
			assert(w == 28 && h == 28);
			int wh = w * h;
			b = new byte[n][wh];
			for (int m = 0; m < n; ++m) {
				bis.read(b[m]);
			}
			System.out.println(w);
			int[] bi = new int[wh];
			f = new float[n][wh];
			IntStream.range(0,  n).forEach(m->{
				IntStream.range(0, wh).forEach(x->{
					bi[x] = (int)b[m][x] & 255;
					f[m][x] = (float)bi[x];
				});
				float bmax = (float)IntStream.range(0,  wh).mapToDouble(x->f[m][x]).max().getAsDouble();
				if (normalize) {
					IntStream.range(0, wh).forEach(x->{
						f[m][x] = f[m][x] / bmax;
					
					});
				}
			});
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public int getWidth() {
			// TODO 自動生成されたメソッド・スタブ
			return w;
		}

		@Override
		public int getHeight() {
			// TODO 自動生成されたメソッド・スタブ
			return h;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			return f[k];
		}

		@Override
		public float[][] getContent() {
			return f;
		}

		@Override
		public void setContentDev() {
			// TODO Auto-generated method stub
			// デバイスメモリに転送する
			int wh = w * h;
			devMemArray = new CUdeviceptr[n];
			IntStream.range(0,  n).forEach(m->{
				devMemArray[m] = new CUdeviceptr();
				cuMemAlloc(devMemArray[m], wh * Sizeof.FLOAT);
				cuMemcpyHtoD(devMemArray[m], Pointer.to(f[m]), wh * Sizeof.FLOAT);
			});
			devMem = new CUdeviceptr();
			cuMemAlloc(devMem, n * Sizeof.POINTER);
			cuMemcpyHtoD(devMem, Pointer.to(devMemArray), n * Sizeof.POINTER);
		}

		@Override
		public CUdeviceptr getContentDev(int sample) {
			// TODO Auto-generated method stub
			return devMemArray[sample];
		}

		@Override
		public CUdeviceptr getContentDev() {
			// TODO Auto-generated method stub
			return devMem;
		}
	}

	/**
	 * load MNIST label file
	 * @author minoru
	 *
	 */
	public class MNISTLabel extends ImageLabelSet.BaseLabel {
		public int n;
		public final boolean oneHot;
		public byte b[];
		float bhot[][];
		CUdeviceptr devB;
		CUdeviceptr[] devBhotArray;
		CUdeviceptr devBhot;
		
		MNISTLabel(boolean oneHot) {
			this.oneHot = oneHot;
		}
		
		public void finalize() {
			if (devB != null) {
				cuMemFree(devB);
				devB = null;
			}
			if (devBhot != null) {
				IntStream.range(0,  n).forEach(i->{
					cuMemFree(devBhotArray[i]);
				});
				cuMemFree(devBhot);
			}
		}
		
		/**
		 * return true if two parameters are equal
		 * @param i
		 * @param j
		 * @return
		 */
		private  float getPattern(int i, int j) {
			return (i == j) ? 1.0f : 0.0f;
		}

		/**
		 * label file loader
		 * @param bis
		 * @throws IOException
		 */
		public void loader(BufferedInputStream bis) throws IOException {
			// magic number を無視
			MNIST.readInt(bis);
			// 数
			n = MNIST.readInt(bis);
			b = new byte[n];
			bis.read(b);
			if (oneHot) {
				bhot = new float[n][10];
				for (int i = 0; i < n; ++i) {
					for (int j = 0; j < 10; ++j) {
						bhot[i][j] = getPattern(b[i], j);
					}
				}
			}
		}

		@Override
		public int getQuantity() {
			// TODO 自動生成されたメソッド・スタブ
			return n;
		}

		@Override
		public float[] getContent(int k) {
			// TODO 自動生成されたメソッド・スタブ
			return bhot[k];
		}

		@Override
		public int getOutputCount() {
			// TODO 自動生成されたメソッド・スタブ
			return 10;	// 0～9
		}

		@Override
		public float[][] getContent() {
			// TODO 自動生成されたメソッド・スタブ
			return bhot;
		}

		@Override
		public void setContentDev() {
			// TODO Auto-generated method stub
			devB = new CUdeviceptr();
			cuMemAlloc(devB, Sizeof.BYTE * n);
			cuMemcpyHtoD(devB, Pointer.to(b), Sizeof.BYTE * n);
			if (bhot != null) {
				devBhotArray = new CUdeviceptr[n];
				IntStream.range(0,  n).forEach(i->{
					devBhotArray[i] = new CUdeviceptr();
					cuMemAlloc(devBhotArray[i], Sizeof.FLOAT * 10);
					cuMemcpyHtoD(devBhotArray[i], Pointer.to(bhot[i]), Sizeof.FLOAT * 10);
				});
				devBhot = new CUdeviceptr();
				cuMemAlloc(devBhot, Sizeof.POINTER * n);
				cuMemcpyHtoD(devBhot, Pointer.to(devBhotArray), Sizeof.POINTER * n);
			}
		}

		@Override
		public CUdeviceptr getContentDev(int sample) {
			// TODO Auto-generated method stub
			return devBhotArray[sample];
		}

		@Override
		public CUdeviceptr getContentDev() {
			// TODO Auto-generated method stub
			return devBhot;
		}
	}


	/**
	 * private constructor
	 */
	MNIST(boolean normalize, boolean oneHot) {
		image = new MNISTImage(normalize);
		label = new MNISTLabel(oneHot);
	}

	/**
	 * read 32bit integer (big endian)
	 * @param bis
	 * @return
	 * @throws IOException
	 */
	static int readInt(BufferedInputStream bis) throws IOException {
		byte[] b = new byte[4];
		bis.read(b);
		return ByteBuffer.wrap(b).getInt();
	}

	/**
	 * load the specific MNIST file set
	 *
	 * @param imagef
	 * @param labelf
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load(final String imagef, final String labelf, boolean normalize,
			boolean oneHot) throws IOException {
		MNIST mnist = new MNIST(normalize, oneHot);
		BufferedInputStream bis = null;
		try {
			bis = new BufferedInputStream(new FileInputStream(imagef));
			mnist.image.loader(bis);
			bis.close();
			bis = new BufferedInputStream(new FileInputStream(labelf));
			mnist.label.loader(bis);
			
			// デバイスメモリに展開
			mnist.image.setContentDev();
			mnist.label.setContentDev();
		} finally {
			if (bis != null) bis.close();
		}
		return mnist;
	}

	/**
	 * load 60000 samples for training
	 *
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load_train(boolean normalize, boolean one_hot) throws IOException{
		return load(train_images, train_labels, normalize, one_hot);
	}

	/**
	 * load 10000 samples for test
	 *
	 * @param normalize
	 * @param one_hot
	 * @return
	 * @throws IOException
	 */
	public static MNIST load_test(boolean normalize, boolean one_hot) throws IOException{
		return load(t10k_images, t10k_labels, normalize, one_hot);
	}

	public static void main(String[] args) throws IOException {
		MNIST train = MNIST.load_train(true, false);
		MNIST test = MNIST.load_test(true, false);
		System.out.printf("train: %d samples; %d width %d height",
				train.image.getQuantity(), train.image.getWidth(), train.image.getHeight());
		System.out.printf(" %d labels:",
				train.label.getQuantity());
		System.out.printf("test: %d samples; %d width %d height",
				test.image.getQuantity(), test.image.getWidth(), test.image.getHeight());
		System.out.printf(" %d labels:",
				test.label.getQuantity());
	}
}
