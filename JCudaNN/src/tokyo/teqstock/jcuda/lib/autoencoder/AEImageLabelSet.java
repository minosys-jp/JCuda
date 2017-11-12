package tokyo.teqstock.jcuda.lib.autoencoder;

import java.io.BufferedInputStream;
import java.io.IOException;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuMemAlloc;
import static jcuda.driver.JCudaDriver.cuMemFree;
import static jcuda.driver.JCudaDriver.cuMemcpyDtoD;
import static jcuda.driver.JCudaDriver.cuMemcpyHtoD;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUdeviceptr;
import tokyo.teqstock.jcuda.lib.ImageLabelSet;

/**
 * ImageLabelSet for AutoEncoder
 * 
 * @author minoru
 *
 */
public class AEImageLabelSet extends ImageLabelSet {
	CUdeviceptr devOut;
	CUdeviceptr[] devOutArray;
	protected final int wh, n;
	
	/**
	 * constructor
	 * 
	 * @param devImageArray
	 * @param wh
	 */
	public AEImageLabelSet(int n, int wh) {
		this.n = n;
		this.wh = wh;
	}

	/**
	 * copy sample images
	 * 
	 * @param devImage
	 */
	public void setContentDev(CUdeviceptr devOut, CUdeviceptr[] devOutArray) {
		this.devOut = devOut;
		this.devOutArray = devOutArray;
		if (image == null) {
			image = new AEImage(devOut, devOutArray, wh);
		}
		if (label == null) {
			label = new AELabel(devOut, devOutArray, wh);
		}
	}
	
	/**
	 * AutoEncoder Image class
	 * @author minoru
	 *
	 */
	public class AEImage extends ImageLabelSet.BaseImage {
		private final CUdeviceptr devOut;
		private final CUdeviceptr[] devOutArray;
		private final int wh;
		
		public AEImage(CUdeviceptr devOut, CUdeviceptr[] devOutArray, int wh) {
			this.devOut = devOut;
			this.devOutArray = devOutArray;
			this.wh = wh;
		}
		
		@Override
		public void loader(BufferedInputStream bis) throws IOException {
			// TODO Auto-generated method stub
			
		}
		
		@Override
		public int getQuantity() {
			// TODO Auto-generated method stub
			return devOutArray.length;
		}
		
		@Override
		public float[] getContent(int k) {
			// TODO Auto-generated method stub
			return null;
		}
		
		@Override
		public void setContentDev() {
			// TODO Auto-generated method stub
		}
		
		@Override
		public CUdeviceptr getContentDev(int sample) {
			// TODO Auto-generated method stub
			return devOutArray[sample];
		}
		
		@Override
		public CUdeviceptr getContentDev() {
			// TODO Auto-generated method stub
			return devOut;
		}
		
		@Override
		public float[][] getContent() {
			// TODO Auto-generated method stub
			return null;
		}
		
		@Override
		public int getWidth() {
			// TODO Auto-generated method stub
			return wh;
		}
		
		@Override
		public int getHeight() {
			// TODO Auto-generated method stub
			return 1;
		}
		
	}
	
	/**
	 * AutoEncoder Label class
	 * 
	 * @author minoru
	 *
	 */
	public class AELabel extends ImageLabelSet.BaseLabel {
		private final CUdeviceptr devOut;
		private final CUdeviceptr[] devOutArray;
		private final int wh;
		
		public AELabel(CUdeviceptr devOut, CUdeviceptr[] devOutArray, int wh) {
			this.devOut = devOut;
			this.devOutArray = devOutArray;
			this.wh = wh;
		}
		
		public void finalize() {
		}
		
		@Override
		public void loader(BufferedInputStream bis) throws IOException {
			// TODO Auto-generated method stub
			
		}

		@Override
		public int getQuantity() {
			// TODO Auto-generated method stub
			return devOutArray.length;
		}

		@Override
		public float[] getContent(int k) {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public void setContentDev() {
			// TODO Auto-generated method stub
			
		}

		@Override
		public CUdeviceptr getContentDev(int sample) {
			// TODO Auto-generated method stub
			return devOutArray[sample];
		}

		@Override
		public CUdeviceptr getContentDev() {
			// TODO Auto-generated method stub
			return devOut;
		}

		@Override
		public float[][] getContent() {
			// TODO Auto-generated method stub
			return null;
		}

		@Override
		public int getOutputCount() {
			// TODO Auto-generated method stub
			return wh;
		}
	}
}
