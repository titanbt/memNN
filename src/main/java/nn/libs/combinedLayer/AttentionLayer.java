package nn.libs.combinedLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

import nn.libs.MultiConnectLayer;
import nn.libs.NNInterface;
import nn.libs.SoftmaxLayer;

public class AttentionLayer implements NNInterface{

	// we require the inputlength is equals to the length of each context vector
	// actually this is not necessary for connectlineartanh, only required for connectInnerProduct
	
	public double[] inputG;
	public double[] contextGs;
	
	public List<NNInterface> attentionCells;
	MultiConnectLayer attentionConnect;
	public SoftmaxLayer attentionSoftmax;
	
	public double[] output;
	public double[] outputG;
	
	int inputLength;
	int contextNum;
	
	/**
	 * We let the input length to be equal with each context vector
	 * @param seedAttentionCell
	 * @param xInputs: the input vector, e.g. the target vec 
	 * @param contextVecs: the concatenation of context vectors
	 * @param xContextNum: the number of context words/sentences
	 * @throws Exception
	 */
	public AttentionLayer(
			NNInterface seedAttentionCell,
			int xInputLength,
			int xContextNum) throws Exception
	{
		inputLength = xInputLength;
		contextNum = xContextNum;
		
		if(( (double[])seedAttentionCell.getInput(0) ).length != inputLength ||
				( (double[])seedAttentionCell.getInput(1) ).length != inputLength)
		{
			throw new Exception("Connection input length does not match.");
		}

		attentionCells = new ArrayList<NNInterface>();
		
		int[] contextLengths = new int[contextNum];
		Arrays.fill(contextLengths, 1);
		attentionConnect = new MultiConnectLayer(contextLengths);
		
		for(int i = 0; i < contextNum; i++)
		{
			NNInterface tmpCell = 
				(NNInterface) seedAttentionCell.cloneWithTiedParams();

			tmpCell.link(attentionConnect, i);
			attentionCells.add(tmpCell);
		}

		attentionSoftmax = new SoftmaxLayer(contextNum);
		attentionConnect.link(attentionSoftmax);
		
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
		for(int i = 0; i < contextNum; i++)
		{
			System.arraycopy(xInput, 0, (double[]) attentionCells.get(i).getInput(0), 0, inputLength);
			
			System.arraycopy(xContextVecs, i * inputLength, (double[]) attentionCells.get(i).getInput(1), 0, inputLength);
		}
		
		for(NNInterface layer: attentionCells)
		{
			layer.forward();
		}
		attentionConnect.forward();
		attentionSoftmax.forward();
		
		for(int i = 0; i < inputLength; i++)
			output[i] = 0;
		
		for(int i = 0; i < contextNum; i++)
		{
			// output length is equal to input.length
			for(int j = 0; j < inputLength; j++)
			{
				output[j] += attentionSoftmax.output[i] * 
					( (double[]) attentionCells.get(i).getInput(1) )[j];
			}
		}
	}
	
	public void backward() {
		for(int i = 0; i < contextNum; i++)
		{
			// output length is equal to input.length
			for(int j = 0; j < inputLength; j++)
			{
				attentionSoftmax.outputG[i] += outputG[j] 
				  * ( (double[]) attentionCells.get(i).getInput(1) )[j];
			}
		}
		
		attentionSoftmax.backward();
		attentionConnect.backward();
		
		for(int i = 0; i < contextNum; i++)
		{
			attentionCells.get(i).backward();
		}
		
		int k = 0;
		for(int i = 0; i < contextNum; i++)
		{
			for(int j = 0; j < inputLength; j++)
			{
				((double[]) attentionCells.get(i).getInputG(1) )[j] += 
						outputG[j] * attentionSoftmax.output[i];
				
				contextGs[k] += ((double[]) attentionCells.get(i).getInputG(1))[j];
				k++;
			}
		}
		
		// calculate for inputG
		for(int i = 0; i < contextNum; i++)
		{
			NNInterface layer = attentionCells.get(i);
			for(int j = 0; j < inputLength; j++)
			{
				inputG[j] += ( (double[]) layer.getInputG(0) )[j];
			}
		}
	}

	public void update(double learningRate) {
		for(NNInterface layer: attentionCells)
			layer.update(learningRate);
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		for(NNInterface layer: attentionCells)
			layer.updateAdaGrad(learningRate, batchsize);
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		for(NNInterface layer: attentionCells)
		{
			layer.clearGrad();
		}
		
		attentionConnect.clearGrad();
		attentionSoftmax.clearGrad();
		attentionCells.clear();
		
		Arrays.fill(inputG, 0);
		Arrays.fill(contextGs, 0);
		Arrays.fill(outputG, 0);
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
		output = nextI;
		outputG = nextIG;
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
		return output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		return null;
	}

}
