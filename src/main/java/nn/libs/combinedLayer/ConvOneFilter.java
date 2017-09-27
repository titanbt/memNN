package nn.libs.combinedLayer;

import nn.libs.AverageLayer;
import nn.libs.MultiConnectLayer;
import nn.libs.NNInterface;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ConvOneFilter implements NNInterface{
	List<ConnectLinearTanh> LLTlist;
	MultiConnectLayer connect;
	public AverageLayer average;

	int linkId;
	int inputLength;
	int outputLength;
	int contextNum;
	int windowSizeLookup;

	public ConvOneFilter()
	{
	}

	public ConvOneFilter(
			ConnectLinearTanh seedLLT,
			int xContextNum,
			int xInputLength,
			int xWindowSizeLookup
		) throws Exception 
	{
		contextNum = xContextNum;
		inputLength = xInputLength;

		windowSizeLookup = xWindowSizeLookup;
		LLTlist = new ArrayList<ConnectLinearTanh>();
		
		for(int i = 0; i < contextNum - windowSizeLookup + 1; i++)
		{
			ConnectLinearTanh tmpLLT = (ConnectLinearTanh) seedLLT.cloneWithTiedParams();
			LLTlist.add(tmpLLT);
		}
		
		int[] connectInputLengths = new int[LLTlist.size()];
		Arrays.fill(connectInputLengths, LLTlist.get(0).linear1.outputLength);

		connect = new MultiConnectLayer(connectInputLengths);
		for(int k = 0; k < LLTlist.size(); k++)
		{
			LLTlist.get(k).link(connect, k);
		}
		
		average = new AverageLayer(connect.outputLength, LLTlist.get(0).linear1.outputLength);
		connect.link(average);
		
		linkId = 0;
		outputLength = average.outputLength;
	}
	
	public void randomize(Random r, double min, double max) {
		
	}

	public void forward() {
		System.err.println("do not call this function");
	}

	public double[] calAverageVecs(double[][] contextVecs, int inputLength) {
		double[] averageVec = new double[inputLength];
		for (int i = 0; i < contextVecs.length; i++) {
			for (int j = 0; j < averageVec.length; j++) {
				averageVec[j] += contextVecs[i][j];
			}
		}
		for (int j = 0; j < averageVec.length; j++) {
			averageVec[j] = averageVec[j] / contextVecs.length;
		}
		return averageVec;
	}
	public void forward(double[] xInput, double[] xContextVecs) {
		double[][] contextVecs = new double[windowSizeLookup][inputLength];
		for (int i = 0; i < contextNum - windowSizeLookup + 1; i++) {
			System.arraycopy(xInput, 0, (double[]) LLTlist.get(i).getInput(0), 0, inputLength);
			for (int j = 0; j < windowSizeLookup; j++) {
				System.arraycopy(xContextVecs, (i + j) * inputLength, contextVecs[j], 0, inputLength);
			}
			System.arraycopy(calAverageVecs(contextVecs, inputLength), 0, (double[]) LLTlist.get(i).getInput(1), 0, inputLength);
		}

		for(ConnectLinearTanh layer: LLTlist)
		{
			layer.forward();
		}
		connect.forward();
		average.forward();
	}

	public void backward() {
		average.backward();
		connect.backward();
		for(ConnectLinearTanh layer: LLTlist)
		{
			layer.backward();
		}
	}

	public void update(double learningRate) {
		for(ConnectLinearTanh layer: LLTlist)
		{
			layer.update(learningRate);
		}
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		for(ConnectLinearTanh layer: LLTlist)
		{
			layer.clearGrad();
		}
		
		connect.clearGrad();
		average.clearGrad();
		
		LLTlist.clear();
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
		return LLTlist.get(id).getInputG(1);
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return average.outputG;
	}

	public Object cloneWithTiedParams() {
		return null;
	}
}
