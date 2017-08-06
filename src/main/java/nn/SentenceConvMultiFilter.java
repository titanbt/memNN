package nn;

import nn.libs.AverageLayer;
import nn.libs.MultiConnectLayer;
import nn.libs.NNInterface;
import nn.libs.combinedLayer.LookupLinearTanh;

import java.util.Random;

public class SentenceConvMultiFilter implements NNInterface{
	
	SentenceConvOneFilter filter1;
	SentenceConvOneFilter filter2;
	SentenceConvOneFilter filter3;
	
	MultiConnectLayer connect;
	AverageLayer average;
	
	int linkId;
	int outputLength;
	
	public SentenceConvMultiFilter()
	{
	}
	
	public SentenceConvMultiFilter(
			int[] wordIds,
			LookupLinearTanh seedLLT1,
			LookupLinearTanh seedLLT2,
			LookupLinearTanh seedLLT3
		) throws Exception 
	{
		filter1 = new SentenceConvOneFilter(wordIds, seedLLT1);
		filter2 = new SentenceConvOneFilter(wordIds, seedLLT2);
		filter3 = new SentenceConvOneFilter(wordIds, seedLLT3);

		outputLength = filter1.outputLength;
		
		connect = new MultiConnectLayer(new int[]{filter1.outputLength, 
		          filter2.outputLength, filter3.outputLength});
		
		average = new AverageLayer(connect.outputLength, outputLength);
		
		filter1.link(connect, 0);
		filter2.link(connect, 1);
		filter3.link(connect, 2);
		
		connect.link(average);
	}

	public void randomize(Random r, double min, double max) {
		
	}

	public void forward() {
		filter1.forward();
		filter2.forward();
		filter3.forward();
		connect.forward();
		average.forward();
	}

	public void backward() {
		average.backward();
		connect.backward();
		filter3.backward();
		filter2.backward();
		filter1.backward();
	}

	public void update(double learningRate) {
		filter1.update(learningRate);
		filter2.update(learningRate);
		filter3.update(learningRate);
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		filter1.clearGrad();
		filter2.clearGrad();
		filter3.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != average.output.length || nextIG.length != average.outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		average.output = nextI;
		average.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return average.output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return average.outputG;
	}

	public Object cloneWithTiedParams() {
		return null;
	}
}
