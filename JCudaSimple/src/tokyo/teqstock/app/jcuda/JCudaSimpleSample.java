package tokyo.teqstock.app.jcuda;

/*
 * JCuda - Java bindings for NVIDIA CUDA driver and runtime API
 * http://www.jcuda.org
 *
 * Copyright 2011 Marco Hutter - http://www.jcuda.org
 */

import static jcuda.driver.JCudaDriver.*;

import java.io.*;
import java.util.Random;
import java.util.stream.IntStream;

import jcuda.*;
import jcuda.driver.*;

/**
 * （たぶん）一番簡単な JCuda の説明プログラム
 */
public class JCudaSimpleSample
{
	private static final String CUFILENAME = "JCudaSimpleKernel.cu";
	private static final String SIMPLEKERNEL = "simpleKernel";
	private static final int NTHREADS = 32;
	private static final int XSIZE = 128;
	private static final int YSIZE = 128;
	
	private static CUfunction initJCuda() throws IOException {
        // 例外処理を有効にする
        JCudaDriver.setExceptionsEnabled(true);
        
		// cu ファイルから ptx ファイルを（必要なら）コンパイルしてロード
        String ptxFileName = preparePtxFile(CUFILENAME);

        // CUDA ドライバーを初期化し、最初のデバイスに対するコンテキストを作成する
        cuInit(0);
        CUcontext pctx = new CUcontext();
        CUdevice dev = new CUdevice();
        cuDeviceGet(dev, 0);
        cuCtxCreate(pctx, 0, dev);
        
        // PTX ファイルを読み込む
        CUmodule module = new CUmodule();
        cuModuleLoad(module, ptxFileName);
        
        // 関数名 'sampleKernel' に対する CUDA 関数ポインタを取得する
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, module, SIMPLEKERNEL);
        return function;
	}
	
    public static void main(String args[]) throws IOException
    {
        
    	CUfunction function = initJCuda();

    	// ホスト側のメモリを割当てる
    	// ２次元配列でそれぞれのサイズは XSIZE, YSIZE である
    	Random rnd = new Random();
        float hostInput[][] = new float[XSIZE][YSIZE];
        IntStream.range(0, XSIZE).forEach(i->{
        	IntStream.range(0,  YSIZE).forEach(j->{
        		hostInput[i][j] = rnd.nextFloat();
        	});
        });

        // ２次元配列のデバイスメモリ側の確保と初期化の方法
        // 1. XSIZE 個の hostDevicePointers を定義する。（YSIZE * Sizeof.FLOAT バイトある）
        // 2. hostDevicePointers にホスト側のメモリの内容をコピーする
        // 3. deviceInput に XSIZE * Sizeof.POINTER バイトのメモリを用意する
        // 4. hostDevicePointers をデバイスメモリにコピーする

        // hostDevicePointers を定義する
        CUdeviceptr[] hostDevicePointers = new CUdeviceptr[XSIZE];
        
        // XSIZE 回のループ
        IntStream.range(0, XSIZE).forEach(i->{
        	// CUdeviceptr の作成と Y 方向のデバイスメモリ確保
        	hostDevicePointers[i] = new CUdeviceptr();
        	cuMemAlloc(hostDevicePointers[i], YSIZE * Sizeof.FLOAT);
        	
        	// デバイスメモリにホストメモリの内容をコピー（Y方向１列分をコピー）
            cuMemcpyHtoD(hostDevicePointers[i],
                    Pointer.to(hostInput[i]), YSIZE * Sizeof.FLOAT);
        });

        // 配列ポインタのためのデバイスメモリを確保し、コピーする
        CUdeviceptr deviceInput = new CUdeviceptr();
        cuMemAlloc(deviceInput, XSIZE * Sizeof.POINTER);
        cuMemcpyHtoD(deviceInput, Pointer.to(hostDevicePointers),
            XSIZE * Sizeof.POINTER);

        // 結果（１次元配列）の確保
        CUdeviceptr deviceOutput = new CUdeviceptr();
        cuMemAlloc(deviceOutput, XSIZE * Sizeof.FLOAT);

        // カーネル起動パラメータの編集
        Pointer kernelParams = Pointer.to(
        			Pointer.to(deviceInput),
        			Pointer.to(new int[]{XSIZE}),
        			Pointer.to(new int[]{YSIZE}),
        			Pointer.to(deviceOutput)
        		);

        // カーネルの呼び出し
        cuLaunchKernel(function, 
            (XSIZE + NTHREADS - 1) / NTHREADS, 1, 1,	// Grid の次元数 
            NTHREADS, 1, 1, 							// Block の次元数
            0, null,									// 共有メモリサイズおよび stream （現在はデフォルトを指定） 
            kernelParams, null							// カーネルパラメータおよび拡張パラメータ（常に null を指定する）
        ); 
        cuCtxSynchronize();		// 結果を取得するまで待機

        // 出力に対応するホスト側のメモリを割り当てる
        float[] hostOutput = new float[XSIZE];
        cuMemcpyDtoH(Pointer.to(hostOutput), deviceOutput, XSIZE * Sizeof.FLOAT);
        
        // CPU と GPU の結果を照合する
        // CPU で期待値を計算する
        float[] expected = new float[XSIZE];
        IntStream.range(0, YSIZE).forEach(i->{
        	expected[i] = (float) IntStream.range(0, YSIZE).mapToDouble(j->hostInput[i][j]).sum();
        });
        
        // 差が一定値以下の添字のみ残し、その出現数をカウントする
        int count = (int) IntStream.range(0, XSIZE).filter(i->Math.abs(expected[i] - hostOutput[i]) < 1e-4).count();
        
        // 成功率を出力する
        double rate = (double)count / (double) XSIZE * 100.0;
        System.out.printf("Test success rate=%.2f %%\n", rate);

        // 後処理
        IntStream.range(0,  XSIZE).forEach(i->{
        	cuMemFree(hostDevicePointers[i]);
        });
        cuMemFree(deviceInput);
        cuMemFree(deviceOutput);
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
