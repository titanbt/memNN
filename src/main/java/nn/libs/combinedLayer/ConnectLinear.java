package nn.libs.combinedLayer;

import java.util.Random;

import nn.libs.*;

public class ConnectLinear implements NNInterface {

	public MultiConnectLayer connect;
	public LinearLayer linear;
	
	public ConnectLinear() {
		// TODO Auto-generated constructor stub
	}
	
	public ConnectLinear(int xConnectInputLength1,
			int xConnectInputLength2,
			int hiddenLength) throws Exception
	{
		connect = new MultiConnectLayer(new int[]{xConnectInputLength1, xConnectInputLength2});
		linear = new LinearLayer(connect.outputLength, hiddenLength);
		
		connect.link(linear);
	}

	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		linear.randomize(r, min, max);
	}

	public void forward() {
		// TODO Auto-generated method stub
		connect.forward();
		linear.forward();
	}

	public void backward() {
		// TODO Auto-generated method stub
		linear.backward();
		connect.backward();
	}

	public void update(double learningRate) {
		// TODO Auto-generated method stub
		linear.update(learningRate);
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		connect.clearGrad();
		linear.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != linear.output.length 
				|| nextIG.length != linear.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		linear.output = nextI;
		linear.outputG = nextIG;
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
		return linear.output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return connect.getInputG(id);
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return linear.outputG;
	}

	public Object cloneWithTiedParams() {
		ConnectLinear clone = new ConnectLinear();
		
		clone.connect = (MultiConnectLayer) connect.cloneWithTiedParams();
		clone.linear = (LinearLayer) linear.cloneWithTiedParams();
		
		try {
			clone.connect.link(clone.linear);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		return clone;
	}

}
