package tokyo.teqstock.jcuda.lib.autoencoder;

import java.util.Map;
import java.util.stream.IntStream;

import jcuda.driver.CUdeviceptr;
import jcuda.driver.CUfunction;
import tokyo.teqstock.jcuda.lib.NeuralNet;
import tokyo.teqstock.jcuda.lib.SimpleNet;

/**
 * 最上層の sample ノードから leaf まで forward した結果を保持する
 * @author minoru
 *
 */
public class AENodeHierarchy {
	private final AEImageLabelSet[] aes;
	private final int nsamples;
	private final int[] nodes;
	private CUdeviceptr devImage, devLabel;
	
	public AENodeHierarchy(Map<String, CUfunction> fMapper, int nsamples, int[] nodes) {
		this.nsamples = nsamples;
		this.nodes = nodes;
		aes = new AEImageLabelSet[nodes.length - 1];
		IntStream.range(0, nodes.length - 1).forEach(i->{
			if (i == aes.length - 1) {
				// leaf node
				aes[i] = new AENodeImageLabelSet(fMapper, nsamples, nodes[i], nodes[i + 1]);
			} else {
				// intermediate node
				aes[i] = new AEImageLabelSet(nsamples, nodes[i]);
			}
		});
	}
	
	public void setContentDev(CUdeviceptr devImage, CUdeviceptr[] devImageArray, CUdeviceptr devLabel) {
		this.devImage = devImage;
		this.devLabel = devLabel;

		if (isLeaf(0)) {
			// leaf node
			AENodeImageLabelSet leaf = (AENodeImageLabelSet)aes[0];
			leaf.setContentDev(devImage, devLabel);
		} else {
			// intermediate nodes
			aes[0].setContentDev(devImage, devImageArray);
		}
	}
	
	public boolean isLeaf(int index) {
		return index == aes.length - 1;
	}
	
	public int getNsamples() {
		return nsamples;
	}
	
	public AEImageLabelSet getILS(int layer) {
		return aes[layer];
	}
	
	public AEImageLabelSet forward(NeuralNet nn, int layer) {
		SimpleNet sn = nn.neurons[layer];
		sn.forward(aes[layer].devOut);
		if (isLeaf(layer)) {
			// leaf node; connecting to the last
			AENodeImageLabelSet leaf = (AENodeImageLabelSet)aes[layer + 1];
			leaf.setContentDev(sn.devOutz, devLabel);
		} else {
			// intermediate node
			aes[layer + 1].setContentDev(sn.devOutz, sn.devOutz2D);
		}
		return aes[layer + 1];
	}
}
