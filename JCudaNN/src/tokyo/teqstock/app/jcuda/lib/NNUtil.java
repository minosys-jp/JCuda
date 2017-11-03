package tokyo.teqstock.app.jcuda.lib;

import java.util.Map;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;

public class NNUtil {
	private static final String[] KERNELS = {
			// SimpleNet.java
			"clear1D", "clear2D", "calc_linear", "calc_output", "loss_derivative",
			"calc_deriv_b_kernel", "calc_deriv_w_kernel",
			
			// NeuralNet.java
			"vec_add_1d", "vec_add_2d", "learn_1d", "learn_2d"
	};
	public static void createMapper(Map<String, CUfunction> fMapper, CUmodule module) {
		IntStream.range(0,  KERNELS.length).forEach(i->{
			String key = KERNELS[i];
			CUfunction f = new CUfunction();
			cuModuleGetFunction(f, module, key);
			fMapper.put(key, f);
		});
	}
}
