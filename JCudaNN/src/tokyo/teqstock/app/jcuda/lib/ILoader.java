package tokyo.teqstock.app.jcuda.lib;

import java.io.BufferedInputStream;
import java.io.IOException;
import jcuda.driver.CUdeviceptr;

/**
 * Loader インタフェース
 * @author minoru
 *
 */
public interface ILoader {
	public void loader(BufferedInputStream bis) throws IOException;
	public int getQuantity();
	public float[] getContent(int k);
	public void setContentDev();
	public CUdeviceptr getContentDev(int sample);
	public CUdeviceptr getContentDev();
	public float[][] getContent();
}
