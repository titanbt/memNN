package nn.libs;

import java.util.Random;

public class AverageLayer implements NNInterface{

	public int inputLength;
    public int outputLength;

    public double[] input;
    public double[] inputG;
    
    public double[] output;
    public double[] outputG;

    public int linkId;
	
    public AverageLayer()
    {
    }

    public AverageLayer(int xInputLength, int xOutputLength)
    {
    	this(xInputLength, xOutputLength, 0);
    }
    
    public AverageLayer(int xInputLength, int xOutputLength, int xLinkId)
    {
    	inputLength = xInputLength;
    	outputLength = xOutputLength;
		linkId = xLinkId;
		input = new double[inputLength];
		inputG = new double[inputLength];
		output = new double[outputLength];
		outputG = new double[outputLength];
    }
    
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	public void forward() {
		// TODO Auto-generated method stub
		for(int i = 0; i < outputLength; i++)
		{
			output[i] = 0;
		}
		
		int K = inputLength / outputLength;
		for(int j = 0; j < K; j++)
		{
			for(int i = 0; i < outputLength; i++)
			{
				output[i] += input[j * outputLength + i];
			}
		}
		
		for(int i = 0; i < outputLength; i++)
		{
			output[i] = output[i] / K;
		}
	}

	public void backward() {
		// TODO Auto-generated method stub
		int K = inputLength / outputLength;
		for(int j = 0; j < K; j++)
		{
			for(int i = 0; i < outputLength; i++)
			{
				inputG[i + j * outputLength] = outputG[i] / K;
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
			outputG[i] = 0;
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
		
		if(nextI.length != output.length || nextIG.length != outputG.length)
		{
			throw new Exception("The Lengths of linked layers do not match.");
		}
		output = nextI;
		outputG = nextIG;
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
		return output;
	}

	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		AverageLayer clone = new AverageLayer(inputLength, outputLength, linkId);
		return clone;
	}

}
