package nn.libs.combinedLayer;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

import nn.libs.*;

public class ConnectInnerProduct implements NNInterface {

	public MultiConnectLayer connect;
	public InnerProductLayer innerProduct;
	
	int length;
	
	public ConnectInnerProduct() {
		// TODO Auto-generated constructor stub
	}
	
	public void dumpToStream(Writer bw) throws IOException
    {
    	bw.write(length + "\n");
    	connect.dumpToStream(bw);
    	innerProduct.dumpToStream(bw);
    }
    
    public static ConnectInnerProduct loadFromStream(BufferedReader br) 
    		throws Exception 
    {
    	int _length = Integer.parseInt(br.readLine());
    	MultiConnectLayer _connect = MultiConnectLayer.loadFromStream(br);
    	InnerProductLayer _innerProduct = InnerProductLayer.loadFromStream(br);
    	
    	ConnectInnerProduct layer = new ConnectInnerProduct();
    	
    	layer.length = _length;
    	layer.connect = (MultiConnectLayer) _connect.cloneWithTiedParams();
    	layer.innerProduct = (InnerProductLayer) _innerProduct.cloneWithTiedParams();
    	
    	layer.connect.link(layer.innerProduct);
    	
    	return layer;
    }
	
	public ConnectInnerProduct(int xConnectInputLength1,
			int xConnectInputLength2) throws Exception
	{
		if(xConnectInputLength1 != xConnectInputLength2)
		{
			throw new Exception("xConnectInputLength1 != xConnectInputLength2, "
					+ "you could use ConnectLinearTanh as an alternative.");
			
		}
		length = xConnectInputLength1;
		connect = new MultiConnectLayer(new int[]{length, length});
		innerProduct = new InnerProductLayer(length);
		
		connect.link(innerProduct);
	}
	
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
	}

	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		innerProduct.forward();
	}

	public void backward() {
		innerProduct.backward();
		connect.backward();
	}

	public void update(double learningRate) {
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		
	}

	public void clearGrad() {
		connect.clearGrad();
		innerProduct.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[]) nextInput;
		double[] nextIG = (double[]) nextInputG; 
		
		if(nextI.length != innerProduct.output.length 
				|| nextIG.length != innerProduct.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		innerProduct.output = nextI;
		innerProduct.outputG = nextIG;
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
		return innerProduct.output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return innerProduct.outputG;
	}

	public Object cloneWithTiedParams() {
		ConnectInnerProduct clone = new ConnectInnerProduct();
		
		clone.connect = new MultiConnectLayer(new int[]{length, length});
		clone.innerProduct = new InnerProductLayer(length);
		
		try {
			clone.connect.link(clone.innerProduct);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}
