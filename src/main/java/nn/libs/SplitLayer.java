package nn.libs;

import java.util.Random;

public class SplitLayer implements NNInterface{

	public int inputLength;

    public int[] outputLength;

    public double[] input;

    public double[] inputG;

    public double[][] output;

    public double[][] outputG;

    public int linkId;
	
    public int nextId;
    
    public SplitLayer()
    {
    }

    public SplitLayer(int xInputLength, int[] xOutputLength)
    {
    	this(0, xInputLength, xOutputLength);
    }
    
    public SplitLayer(int xLinkId, int xInputLength, int[] xOutputLength)
    {
    	nextId = 0;
    	inputLength = xInputLength;
    	linkId = xLinkId;
    	outputLength = xOutputLength;
    	
    	input = new double[inputLength];
    	inputG = new double[inputLength];
    	
    	output = new double[outputLength.length][];
    	outputG = new double[outputLength.length][];
    	for (int i = 0; i < outputLength.length; ++i)
        {
            output[i] = new double[outputLength[i]];
            outputG[i] = new double[outputLength[i]];
        }
    }

	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	public void forward() {
		// TODO Auto-generated method stub
		int k = 0;
		for(int i = 0; i < outputLength.length; i++)
		{
			for(int j = 0; j < outputLength[i]; j++)
			{
				output[i][j] = input[k];
				k++;
			}
		}
	}

	public void backward() {
		// TODO Auto-generated method stub
		for (int i = 0; i < input.length; ++i)
        {
            inputG[i] = 0;
        }
		
		int k = 0;
		for(int i = 0; i < outputLength.length; i++)
		{
			for(int j = 0; j < outputLength[i]; j++)
			{
				inputG[k] = outputG[i][j];
				k++;
			}
		}
	}

	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	public void clearGrad() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputG.length; i++)
		{
			for(int j = 0; j < outputG[i].length; j++)
			{
				outputG[i][j] = 0;
			}
		}
		
		for(int i = 0; i < inputG.length; i++)
		{
			inputG[i] = 0;
		}
	}

	public void link(NNInterface nextLayer, int id) throws Exception {
		// TODO Auto-generated method stub
		Object nextInputG = nextLayer.getInputG(id);
		Object nextInput = nextLayer.getInput(id);
		
		double[] nextI = (double[])nextInput;
		double[] nextIG = (double[])nextInputG; 
		
		if(nextI.length != outputLength[nextId] 
				|| nextIG.length != outputLength[nextId])
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output[nextId] = nextI;
		outputG[nextId] = nextIG;
		
		nextId++;
	}

	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return input;
	}

	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output[id];
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG[id];
	}

	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		SplitLayer clone = new SplitLayer(linkId, inputLength, outputLength);
		return clone;
	}
}
