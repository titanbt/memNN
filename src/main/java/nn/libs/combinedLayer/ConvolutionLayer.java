package nn.libs.combinedLayer;

import nn.libs.MultiConnectLayer;
import nn.libs.NNInterface;
import nn.libs.AverageLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class ConvolutionLayer implements NNInterface{

	public double[] inputG;
	public double[] contextGs;
	public double[] output;
	public double[] outputG;

	MultiConnectLayer connect;
	AverageLayer average;

	int outputLength;
	int inputLength;
	int contextNum;

	ConvOneFilter filter1;
	ConvOneFilter filter2;
	ConvOneFilter filter3;

	public ConvolutionLayer(
			ConnectLinearTanh seedLLT1,
			ConnectLinearTanh seedLLT2,
			ConnectLinearTanh seedLLT3,
			int xContextNum,
			int xInputLength
			) throws Exception
	{
		contextNum = xContextNum;
		inputLength = xInputLength;

		filter1 = new ConvOneFilter(seedLLT1, contextNum, inputLength, 1);
		filter2 = new ConvOneFilter(seedLLT2, contextNum, inputLength, 2);
		filter3 = new ConvOneFilter(seedLLT3, contextNum, inputLength, 3);

		outputLength = filter1.outputLength;

		connect = new MultiConnectLayer(new int[]{filter1.outputLength, filter2.outputLength, filter3.outputLength});

		average = new AverageLayer(connect.outputLength, outputLength);

		filter1.link(connect, 0);
		filter2.link(connect, 1);
		filter3.link(connect, 2);
		connect.link(average);

		inputG = new double[inputLength];
		output = new double[inputLength];
		outputG = new double[inputLength];
		contextGs = new double[inputLength * contextNum];

	}
	
	public void randomize(Random r, double min, double max) {
		
	}
	
	public void forward() {
		System.err.println("do not call this function");
	}
	
	public void forward(double[] xInput, double[] xContextVecs)
	{
		filter1.forward(xInput, xContextVecs);
		filter2.forward(xInput, xContextVecs);
		filter3.forward(xInput, xContextVecs);
		connect.forward();
		average.forward();
		for(int i = 0; i < inputLength; i++)
			output[i] = 0;

		for(int i = 0; i < contextNum; i++)
		{
			// output length is equal to input.length
			for(int j = 0; j < inputLength; j++)
			{
				output[j] += average.output[i] * ((double[]) filter1.LLTlist.get(i).getInput(1))[j];
			}
		}
	}
	
	public void backward() {
		for(int i = 0; i < contextNum; i++)
		{
			// output length is equal to input.length
			for(int j = 0; j < inputLength; j++)
			{
				average.outputG[i] += outputG[j] * ( (double[]) filter1.LLTlist.get(i).getInput(1) )[j];
			}
		}
		average.backward();
		connect.backward();
		filter3.backward();
		filter2.backward();
		filter1.backward();

		int k = 0;
		for(int i = 0; i < contextNum; i++)
		{
			for(int j = 0; j < inputLength ; j++)
			{
				((double[]) filter1.getInputG(i))[j] += outputG[j] * average.output[i];

				contextGs[k] += ((double[]) filter1.getInputG(i))[j];
				k++;
			}
		}

		for(int i = 0; i < contextNum; i++)
		{
			NNInterface layer = filter1.LLTlist.get(i);
			for(int j = 0; j < inputLength; j++)
			{
				inputG[j] += ((double[]) layer.getInputG(0))[j];
			}
		}
	}

	public void update(double learningRate) {
		filter1.update(learningRate);
		filter2.update(learningRate);
		filter3.update(learningRate);
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
	}

	public void clearGrad() {
		filter1.clearGrad();
		filter2.clearGrad();
		filter3.clearGrad();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		average.output = nextI;
		average.outputG = nextIG;
	}

	public void link(NNInterface nextLayer) throws Exception {
		link(nextLayer, 0);
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
		// TODO Auto-generated method stub
		return null;
	}

}
