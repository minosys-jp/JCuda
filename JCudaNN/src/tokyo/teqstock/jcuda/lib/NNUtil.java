package tokyo.teqstock.jcuda.lib;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.IntStream;

import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.driver.JCudaDriver.cuInit;
import static jcuda.driver.JCudaDriver.cuDeviceGet;
import static jcuda.driver.JCudaDriver.cuCtxCreate;
import static jcuda.driver.JCudaDriver.cuModuleLoad;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;

import jcuda.driver.CUcontext;
import jcuda.driver.CUdevice;
import jcuda.driver.CUfunction;
import jcuda.driver.CUmodule;
import jcuda.driver.JCudaDriver;

public class NNUtil {
	/**
	 * カーネル関数名
	 */
	private static final String[] KERNELS = {
			// SimpleNet.java
			"clear1D", "clear2D", "calc_forward", "loss_derivative",
			"calc_deriv_b_kernel", "calc_deriv_w_kernel",
			
			// NeuralNet.java
			"vec_add_2d", "learn_1d", "learn_2d", "test_sum"
	};
	
	/**
	 * fMapper を作成する
	 * @param fMapper
	 * @param module
	 */
	public static void createMapper(Map<String, CUfunction> fMapper, CUmodule module) {
		IntStream.range(0,  KERNELS.length).forEach(i->{
			String key = KERNELS[i];
			CUfunction f = new CUfunction();
			cuModuleGetFunction(f, module, key);
			fMapper.put(key, f);
		});
	}
	
	/**
	 * JCudaライブラリの初期化
	 * @param cuFileName
	 * @return
	 * @throws IOException
	 */
	public static Map<String, CUfunction> initJCuda(String cuFileName) throws IOException {
        // 例外処理を有効にする
        JCudaDriver.setExceptionsEnabled(true);
        
		// cu ファイルから ptx ファイルを（必要なら）コンパイルしてロード
        String ptxFileName = preparePtxFile(cuFileName);

        // CUDA ドライバーを初期化し、最初のデバイスに対するコンテキストを作成する
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);
        
        // PTX ファイルを読み込む
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        
        // カーネル関数を展開する
        Map<String, CUfunction> fMapper = new ConcurrentHashMap<String, CUfunction>();
        createMapper(fMapper, module);
        return fMapper;
	}

    /**
     * The extension of the given file name is replaced with "ptx".
     * If the file with the resulting name does not exist, it is 
     * compiled from the given file using NVCC. The name of the 
     * PTX file is returned. 
     * 
     * @param cuFileName The name of the .CU file
     * @return The name of the PTX file
     * @throws IOException If an I/O error occurs
     */
    private static String preparePtxFile(String cuFileName) throws IOException
    {
        int endIndex = cuFileName.lastIndexOf('.');
        if (endIndex == -1)
        {
            endIndex = cuFileName.length()-1;
        }
        String ptxFileName = cuFileName.substring(0, endIndex+1)+"ptx";
        File ptxFile = new File(ptxFileName);
        if (ptxFile.exists())
        {
            return ptxFileName;
        }
        
        File cuFile = new File(cuFileName);
        if (!cuFile.exists())
        {
            throw new IOException("Input file not found: "+cuFileName);
        }
        String modelString = "-m"+System.getProperty("sun.arch.data.model");        
        String command = 
            "nvcc " + modelString + " -ptx "+
            cuFile.getPath()+" -o "+ptxFileName;
        
        System.out.println("Executing\n"+command);
        Process process = Runtime.getRuntime().exec(command);

        String errorMessage = 
            new String(toByteArray(process.getErrorStream()));
        String outputMessage = 
            new String(toByteArray(process.getInputStream()));
        int exitValue = 0;
        try
        {
            exitValue = process.waitFor();
        }
        catch (InterruptedException e)
        {
            Thread.currentThread().interrupt();
            throw new IOException(
                "Interrupted while waiting for nvcc output", e);
        }

        if (exitValue != 0)
        {
            System.out.println("nvcc process exitValue "+exitValue);
            System.out.println("errorMessage:\n"+errorMessage);
            System.out.println("outputMessage:\n"+outputMessage);
            throw new IOException(
                "Could not create .ptx file: "+errorMessage);
        }
        
        System.out.println("Finished creating PTX file");
        return ptxFileName;
    }

    /**
     * Fully reads the given InputStream and returns it as a byte array
     *  
     * @param inputStream The input stream to read
     * @return The byte array containing the data from the input stream
     * @throws IOException If an I/O error occurs
     */
    private static byte[] toByteArray(InputStream inputStream) 
        throws IOException
    {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        byte buffer[] = new byte[8192];
        while (true)
        {
            int read = inputStream.read(buffer);
            if (read == -1)
            {
                break;
            }
            baos.write(buffer, 0, read);
        }
        return baos.toByteArray();
    }
}
