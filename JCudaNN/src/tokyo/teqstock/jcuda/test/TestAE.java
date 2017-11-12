package tokyo.teqstock.jcuda.test;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import jcuda.driver.CUfunction;
import tokyo.teqstock.jcuda.lib.MNIST;
import tokyo.teqstock.jcuda.lib.NNUtil;
import tokyo.teqstock.jcuda.lib.autoencoder.AutoEncoder;

public class TestAE {
	private static final String CUFILENAME = "JCudaNNKernel.cu";
	private static final float LRATE = 0.1f;
	private static final float THRESHOLD = 1.0f;
	private static final int COUNT = 10;
	private static final int[] NODES = { 784, 100, 100, 10 };
	AutoEncoder ae;
	Map<String, CUfunction> fMapper;
	Map<String, Integer> fParamMap;
	MNIST train, test;
	
	public TestAE() {
		fParamMap = new HashMap<String, Integer>();
		fParamMap.put("BATCHSIZE", 128);
		fParamMap.put("NSET", 32);
		fParamMap.put("NSAMPLE1", 128);
		fParamMap.put("NSAMPLE2", 32);
	}
	
	public void prepare() throws IOException {
		train = MNIST.load_train(true, true);
		test = MNIST.load_test(true, true);
		fMapper = NNUtil.initJCuda(CUFILENAME);
		ae = new AutoEncoder(fMapper, fParamMap, NODES, LRATE, THRESHOLD);
	}
	
	public void run() {
		ae.training(train);
		ae.test(test, COUNT);;
	}
	
	public void release() {
		
	}
	
	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		TestAE test = new TestAE();
		test.prepare();
		test.run();
		test.release();
	}
}
