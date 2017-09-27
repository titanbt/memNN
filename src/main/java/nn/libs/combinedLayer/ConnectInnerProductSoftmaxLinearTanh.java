package nn.libs.combinedLayer;

import nn.libs.*;
import sun.plugin.javascript.navig4.Layer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class ConnectInnerProductSoftmaxLinearTanh implements NNInterface {

	public MultiConnectLayer connect;
	public InnerProductLayer innerProduct;
	public SoftmaxLayer softmax;
	public LinearLayer linear1;
	public TanhLayer tanh;

	public ConnectInnerProductSoftmaxLinearTanh() {
		// TODO Auto-generated constructor stub
	}

	public void dumpToStream(Writer bw) throws IOException
    {
    	connect.dumpToStream(bw);
    	innerProduct.dumpToStream(bw);
    	linear1.dumpToStream(bw);
    	tanh.dumpToStream(bw);
    }

    public static ConnectInnerProductSoftmaxLinearTanh loadFromStream(BufferedReader br)
    		throws Exception
    {
    	MultiConnectLayer _connect = MultiConnectLayer.loadFromStream(br);
    	InnerProductLayer _innerProduct = InnerProductLayer.loadFromStream(br);
    	SoftmaxLayer _softmax = SoftmaxLayer.loadFromStream(br);
    	LinearLayer _linear1 = LinearLayer.loadFromStream(br);
    	TanhLayer _tanh = TanhLayer.loadFromStream(br);

    	ConnectInnerProductSoftmaxLinearTanh layer = new ConnectInnerProductSoftmaxLinearTanh();

    	layer.connect = (MultiConnectLayer) _connect.cloneWithTiedParams();
    	layer.innerProduct = (InnerProductLayer) _innerProduct.cloneWithTiedParams();
		layer.softmax = (SoftmaxLayer) _softmax.cloneWithTiedParams();
    	layer.linear1 = (LinearLayer) _linear1.cloneWithTiedParams();
    	layer.tanh = (TanhLayer) _tanh.cloneWithTiedParams();

    	layer.connect.link(layer.innerProduct);
    	layer.innerProduct.link(layer.softmax);
    	layer.softmax.link(layer.linear1);
		layer.linear1.link(layer.tanh);

    	return layer;
    }

	public ConnectInnerProductSoftmaxLinearTanh(int xConnectInputLength1,
												int xConnectInputLength2,
												int hiddenLength) throws Exception
	{
		connect = new MultiConnectLayer(new int[]{xConnectInputLength1, xConnectInputLength2});
		innerProduct = new InnerProductLayer(xConnectInputLength1);
		softmax = new SoftmaxLayer(hiddenLength);
		linear1 = new LinearLayer(1, hiddenLength);
		tanh = new TanhLayer(hiddenLength);
		
		connect.link(innerProduct);
		innerProduct.link(softmax);
		softmax.link(linear1);
		linear1.link(tanh);
	}
	
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear1.randomize(r, min, max);
	}

	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		innerProduct.forward();
		softmax.forward();
		linear1.forward();
		tanh.forward();
	}

	public void backward() {
		// TODO Auto-generated method stub
		tanh.backward();
		linear1.backward();
		softmax.backward();
		innerProduct.backward();
		connect.backward();
	}

	public void update(double learningRate) {
		// TODO Auto-generated method stub
		linear1.update(learningRate);
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		linear1.updateAdaGrad(learningRate, batchsize);
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		connect.clearGrad();
		innerProduct.clearGrad();
		softmax.clearGrad();
		linear1.clearGrad();
		tanh.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != tanh.output.length || nextIG.length != tanh.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		tanh.output = nextI;
		tanh.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, 0);
	}

	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return connect.getInput(id);
	}

	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return tanh.output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return tanh.outputG;
	}

	public Object cloneWithTiedParams() {
		ConnectInnerProductSoftmaxLinearTanh clone = new ConnectInnerProductSoftmaxLinearTanh();
		
		clone.connect = (MultiConnectLayer) connect.cloneWithTiedParams();
		clone.innerProduct = (InnerProductLayer) innerProduct.cloneWithTiedParams();
		clone.softmax = (SoftmaxLayer) softmax.cloneWithTiedParams();
		clone.linear1 = (LinearLayer) linear1.cloneWithTiedParams();
		clone.tanh = (TanhLayer) tanh.cloneWithTiedParams();
		
		try {
			clone.connect.link(clone.innerProduct);
			clone.innerProduct.link(clone.softmax);
			clone.softmax.link(clone.linear1);
			clone.linear1.link(clone.tanh);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}
