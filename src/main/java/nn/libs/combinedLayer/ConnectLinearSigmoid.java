package nn.libs.combinedLayer;

import nn.libs.*;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class ConnectLinearSigmoid implements NNInterface {

	public MultiConnectLayer connect;
	public LinearLayer linear1;
	public SigmoidLayer sigmoid;

	public ConnectLinearSigmoid() {
		// TODO Auto-generated constructor stub
	}

	public void dumpToStream(Writer bw) throws IOException
    {
    	connect.dumpToStream(bw);
    	linear1.dumpToStream(bw);
		sigmoid.dumpToStream(bw);
    }

    public static ConnectLinearSigmoid loadFromStream(BufferedReader br)
    		throws Exception
    {
    	MultiConnectLayer _connect = MultiConnectLayer.loadFromStream(br);
    	LinearLayer _linear1 = LinearLayer.loadFromStream(br);
    	TanhLayer _tanh = TanhLayer.loadFromStream(br);

    	ConnectLinearSigmoid layer = new ConnectLinearSigmoid();

    	layer.connect = (MultiConnectLayer) _connect.cloneWithTiedParams();
    	layer.linear1 = (LinearLayer) _linear1.cloneWithTiedParams();
    	layer.sigmoid = (SigmoidLayer) _tanh.cloneWithTiedParams();

    	layer.connect.link(layer.linear1);
		layer.linear1.link(layer.sigmoid);

    	return layer;
    }

	public ConnectLinearSigmoid(int xConnectInputLength1,
								int xConnectInputLength2,
								int hiddenLength) throws Exception
	{
		connect = new MultiConnectLayer(new int[]{xConnectInputLength1, xConnectInputLength2});
		linear1 = new LinearLayer(connect.outputLength, hiddenLength);
		sigmoid = new SigmoidLayer(hiddenLength);
		
		connect.link(linear1);
		linear1.link(sigmoid);
	}
	
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear1.randomize(r, min, max);
	}

	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		linear1.forward();
		sigmoid.forward();
	}

	public void backward() {
		// TODO Auto-generated method stub
		sigmoid.backward();
		linear1.backward();
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
		linear1.clearGrad();
		sigmoid.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != sigmoid.output.length || nextIG.length != sigmoid.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		sigmoid.output = nextI;
		sigmoid.outputG = nextIG;
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
		return sigmoid.output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return sigmoid.outputG;
	}

	public Object cloneWithTiedParams() {
		ConnectLinearSigmoid clone = new ConnectLinearSigmoid();
		
		clone.connect = (MultiConnectLayer) connect.cloneWithTiedParams();
		clone.linear1 = (LinearLayer) linear1.cloneWithTiedParams();
		clone.sigmoid = (SigmoidLayer) sigmoid.cloneWithTiedParams();
		
		try {
			clone.connect.link(clone.linear1);
			clone.linear1.link(clone.sigmoid);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}
