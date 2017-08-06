package nn.libs;

import java.util.Random;

public class RectifierLayer implements NNInterface{

	public int length;
	public int linkId;
	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	
	public RectifierLayer()
	{
	}

	public RectifierLayer(int xLength)
	{
		this(xLength, 0);
	}
	
	public RectifierLayer(int xLength, int xLinkId)
	{
		length = xLength;
		linkId = xLinkId;
		
		input = new double[length];
		inputG = new double[length];
		output = new double[length];
		outputG = new double[length];
	}
	
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
		
	}

	public void forward() {
		// TODO Auto-generated method stub
		for (int i = 0; i < input.length; ++i)
        {
            if (input[i] >= 0.0)
            {
                output[i] = input[i];
            }
            else
            {
                output[i] = 0;
            }
        }
	}

	public void backward() {
		// TODO Auto-generated method stub
		for (int i = 0; i < input.length; ++i)
        {
            if (input[i] >= 0.0)
            {
                inputG[i] = outputG[i];
            }
            else
            {
                inputG[i] = 0;
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
		RectifierLayer clone = new RectifierLayer(length, linkId);
		return clone;
	}

}
