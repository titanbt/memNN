package nn.libs.combinedLayer;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import nn.libs.*;

public class SentenceHighwayNN implements NNInterface{

	// output is highwayNNList.get(highwayNNList.size() - 1)
	
	List<LookupLayer> lookupList;
	List<HighWayNN> highwayNNList;
	
	int hiddenLength;
	int linkId;
	
	public SentenceHighwayNN()
	{
	}
	
	public SentenceHighwayNN(
			int[] wordIds,
			LookupLayer seedLookup,
			HighWayNN seedHighway
		) throws Exception 
	{
		hiddenLength = seedLookup.embeddingLength;
		
		lookupList = new ArrayList<LookupLayer>();
		
		for(int i = 0; i < wordIds.length; i++)
		{
			LookupLayer tmpLookup = (LookupLayer) seedLookup.cloneWithTiedParams();
			tmpLookup.input[0] = wordIds[i];

			lookupList.add(tmpLookup);
		}
		
		highwayNNList = new ArrayList<HighWayNN>();
		
		for(int i = 0; i < lookupList.size(); i++)
		{
			highwayNNList.add((HighWayNN) seedHighway.cloneWithTiedParams());
		}
		
		// link. important
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).link(highwayNNList.get(i), 0);
			if(i > 0)
			{
				highwayNNList.get(i - 1).link(highwayNNList.get(i), 1);
			}
			else
			{
			}
		}
		
		linkId = 0;
	}
	
	public void randomize(Random r, double min, double max) {
		
	}

	public void forward() {
		// be careful about the order. it is important.
		for(int i = 0; i < lookupList.size(); i++)
		{
			lookupList.get(i).forward();
			highwayNNList.get(i).forward();
		}
	}

	public void backward() {
		// the order is important. Be careful
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			highwayNNList.get(i).backward();
			lookupList.get(i).backward();
		}
	}

	public void update(double learningRate) {
		for(HighWayNN highwayLayer: highwayNNList)
		{
			highwayLayer.update(learningRate);
		}
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = lookupList.size() - 1; i >= 0 ; i--)
		{
			highwayNNList.clear();
			lookupList.clear();
		}
		
		highwayNNList.clear();
		lookupList.clear();
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != highwayNNList.get(highwayNNList.size() - 1).output.length 
				|| nextIG.length != highwayNNList.get(highwayNNList.size() - 1).outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		highwayNNList.get(highwayNNList.size() - 1).output = nextI;
		highwayNNList.get(highwayNNList.size() - 1).outputG = nextIG;
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
		return highwayNNList.get(highwayNNList.size() - 1).output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return null;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return highwayNNList.get(highwayNNList.size() - 1).outputG;
	}

	public Object cloneWithTiedParams() {
		return null;
	}
}
