package tokyo.teqstock.jcuda.lib;

public class ImageLabelSet {
	public abstract class BaseImage implements ILoader {
		public abstract int getWidth();
		public abstract int getHeight();
	}
	public abstract class BaseLabel implements ILoader {
		public abstract int getOutputCount();
	}
	public int getQuantity() {
		return label.getQuantity();
	}
	public BaseImage image;
	public BaseLabel label;
}
