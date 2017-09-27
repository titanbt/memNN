package nn;

import nn.libs.*;
import nn.libs.combinedLayer.AttentionLayer;
import nn.libs.combinedLayer.SentenceLSTM;
import nn.libs.combinedLayer.SimplifiedLSTMLayer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class LSTM_SentenceEntityMemNN_Position implements NNInterface{

	int hiddenLength;
	int linkId;

	double[] targetVec;

	public double[][] outputs;
	public double[][] outputGs;

	NNInterface[] entityTransforms;
	public AttentionLayer[] attentionLayers;
//	List<SentenceLSTM> lstmLayers;
	SentenceLSTM lstmLayer;

	MultiConnectLayer lookupConnect;

	int numberOfHops;
	List<LookupLayer> positionLookupList;
	List<SigmoidLayer> positionSigmoidList;

	public LSTM_SentenceEntityMemNN_Position()
	{

	}

	public LSTM_SentenceEntityMemNN_Position(
			int[] wordIds,
			LookupLayer seedLookup,
			NNInterface seedAttentionCell,
 			NNInterface seedEntityTransform,
			SimplifiedLSTMLayer seedLSTM,
			LookupLayer seedPositionLookup,
			int[] wordPositions,
			double[] xTargetVec,
			int xnumberOfHops
		) throws Exception 
	{
		numberOfHops = xnumberOfHops;
		
		targetVec = new double[xTargetVec.length];
		System.arraycopy(xTargetVec, 0, targetVec, 0, xTargetVec.length);
		
		hiddenLength = seedLookup.embeddingLength;

		int[] lookupConnectLengths = new int[wordIds.length];
		Arrays.fill(lookupConnectLengths, hiddenLength);
		lookupConnect = new MultiConnectLayer(lookupConnectLengths);

		lstmLayer  = new SentenceLSTM(wordIds, seedLookup, seedLSTM);

//		for(int i = 0; i < wordIds.length; i++) {
//			lstmLayer.tanhList.get(i).link(lookupConnect, i);
//		}

		attentionLayers = new AttentionLayer[numberOfHops];
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i] = new AttentionLayer(seedAttentionCell, xTargetVec.length, wordIds.length);
		}
		
		entityTransforms = new NNInterface[numberOfHops]; 
		for(int i = 0; i < numberOfHops; i++)
		{
			entityTransforms[i] = (NNInterface) seedEntityTransform.cloneWithTiedParams();
		}
		
		outputs = new double[numberOfHops][];
		outputGs = new double[numberOfHops][];
		
		for(int i = 0; i < numberOfHops; i++)
		{
			outputs[i] = new double[hiddenLength];
			outputGs[i] = new double[hiddenLength];
		}

		positionLookupList = new ArrayList<LookupLayer>();
		positionSigmoidList = new ArrayList<SigmoidLayer>();

		for(int i = 0; i < wordPositions.length; i++)
		{
			LookupLayer tmpLookup = (LookupLayer) seedPositionLookup.cloneWithTiedParams();
			tmpLookup.input[0] = wordPositions[i];

			SigmoidLayer tmpSigmoid = new SigmoidLayer(seedPositionLookup.embeddingLength);
			tmpLookup.link(tmpSigmoid);

			positionLookupList.add(tmpLookup);
			positionSigmoidList.add(tmpSigmoid);
		}

		linkId = 0;
	}
	

	public void randomize(Random r, double min, double max) {
		
	}

	public void forward() {

		for(int i = 0; i < positionLookupList.size(); i++)
		{
			positionLookupList.get(i).forward();
			positionSigmoidList.get(i).forward();
		}

		lstmLayer.forward();

		for(int i = 0; i < lstmLayer.tanhList.size(); i++)
		{
			for(int j = 0; j < lstmLayer.tanhList.get(i).output.length; j++)
				// we regard lookup-sigmoid as a neural gate
				lookupConnect.input[i][j] = lstmLayer.tanhList.get(i).output[j] * positionSigmoidList.get(i).output[j];
		}

		lookupConnect.forward();
		// i should start from 0 and end until numberOfHops - 1
		for(int i = 0; i < numberOfHops; i++)
		{
			if(i == 0)
			{
				attentionLayers[i].forward(targetVec, lookupConnect.output);
				System.arraycopy(targetVec, 0, 
						(double[]) entityTransforms[i].getInput(0), 0, targetVec.length);
			}
			else
			{
				attentionLayers[i].forward(outputs[i-1], lookupConnect.output);
				System.arraycopy(outputs[i-1], 0, 
						(double[]) entityTransforms[i].getInput(0), 0, outputs[i-1].length);
			}
			
			entityTransforms[i].forward();
			
			for(int j = 0; j < outputs[i].length; j++)
			{
				outputs[i][j] = ((double[]) entityTransforms[i].getOutput(0))[j] 
						+ attentionLayers[i].output[j];
			}
		}
	}

	public void backward() {
		
		// i should start from numberOfHops - 1 and end until 0
		for(int i = numberOfHops - 1; i >=0 ; i--)
		{
			for(int j = 0; j < outputs[i].length; j++)
			{
				attentionLayers[i].outputG[j] += outputGs[i][j];
				((double[]) entityTransforms[i].getOutputG(0) )[j] += outputGs[i][j];
			}
			
			entityTransforms[i].backward();
			attentionLayers[i].backward();
			
			for(int j = 0; j < attentionLayers[i].contextGs.length; j++)
			{
				lookupConnect.outputG[j] += attentionLayers[i].contextGs[j];
			}
			
			if(i > 0)
			{
				for(int j = 0; j < outputGs[i-1].length; j++)
				{
					outputGs[i-1][j] += attentionLayers[i].inputG[j]
							+ ( (double[]) entityTransforms[i].getInputG(0) )[j];
				}
			}
		}

		lookupConnect.backward();
		lstmLayer.backward();

		for(int i = 0; i < positionSigmoidList.size(); i++)
		{
			for(int j = 0; j < positionSigmoidList.get(i).output.length; j++)
				positionSigmoidList.get(i).outputG[j] += lookupConnect.inputG[i][j] * lstmLayer.tanhList.get(i).output[j];
		}

		for(int i = 0; i < positionSigmoidList.size(); i++)
		{
			positionSigmoidList.get(i).backward();
			positionLookupList.get(i).backward();
		}

	}

	public void update(double learningRate) {
		lstmLayer.update(learningRate);
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].update(learningRate);
			entityTransforms[i].update(learningRate);
		}
		for(int i = 0; i < positionLookupList.size(); i++)
		{
			positionLookupList.get(i).update(learningRate);
		}
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		lstmLayer.updateAdaGrad(learningRate,batchsize);
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].updateAdaGrad(learningRate,batchsize);
			entityTransforms[i].updateAdaGrad(learningRate,batchsize);
		}
	}

	public void clearGrad() {
		for(int i = 0; i < numberOfHops; i++)
		{
			attentionLayers[i].clearGrad();
			entityTransforms[i].clearGrad();
			
			Arrays.fill(outputs[i], 0);
			Arrays.fill(outputGs[i], 0);
		}
		lstmLayer.clearGrad();
		lookupConnect.clearGrad();
		for(LookupLayer positionLookup: positionLookupList)
		{
			positionLookup.clearGrad();
		}
		positionLookupList.clear();

		for(SigmoidLayer layer: positionSigmoidList)
		{
			layer.clearGrad();
		}
		positionSigmoidList.clear();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != outputs[numberOfHops - 1].length 
				|| nextIG.length != outputGs[numberOfHops - 1].length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		outputs[numberOfHops - 1] = nextI;
		outputGs[numberOfHops - 1] = nextIG;
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
		return outputs[numberOfHops - 1];
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputGs[numberOfHops - 1];
	}

	public Object cloneWithTiedParams() {
		return null;
	}
}
