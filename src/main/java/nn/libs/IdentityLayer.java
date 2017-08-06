package nn.libs;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Writer;
import java.util.Random;

public class IdentityLayer implements NNInterface{

	public double[] input;
	public double[] inputG;
	public double[] output;
	public double[] outputG;
	public int length;
	public int linkId;
	
	public IdentityLayer() { }

	public void dumpToStream(Writer bw) throws IOException
    {
    	bw.write(length + "\n");
    }
    
    public static IdentityLayer loadFromStream(BufferedReader br) throws IOException 
    {
    	int _length = Integer.parseInt(br.readLine());
    	IdentityLayer layer = new IdentityLayer(_length);
    	return layer;
    }
	
    public IdentityLayer(int xLength)
    {
    	this(xLength, 0);
    }
	
    public IdentityLayer(int xLength, int xLinkId)
    {
    	length = xLength;
    	linkId = xLinkId;
    	input = new double[length];
    	inputG = new double[length];
    	output = new double[length];
    	outputG = new double[length];
    }
    
	@Override
	public void randomize(Random r, double min, double max) {
		// TODO Auto-generated method stub
	}

	@Override
	public void forward() 
	{
		// TODO Auto-generated method stub
		for(int i = 0; i < length; i++)
		{
			output[i] = input[i];
		}
	}

	@Override
	public void backward() 
	{
		// TODO Auto-generated method stub
		for(int i = 0; i < length; i++)
		{
			inputG[i] = outputG[i];
		}
	}

	@Override
	public void update(double learningRate) {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void updateAdaGrad(double learningRate, int batchsize) {
		// TODO Auto-generated method stub
		
	}

	@Override
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

	@Override
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

	@Override
	public void link(NNInterface nextLayer) throws Exception {
		// TODO Auto-generated method stub
		link(nextLayer, linkId);
	}

	@Override
	public Object getInput(int id) {
		// TODO Auto-generated method stub
		return input;
	}

	@Override
	public Object getOutput(int id) {
		// TODO Auto-generated method stub
		return output;
	}

	@Override
	public Object getInputG(int id) {
		// TODO Auto-generated method stub
		return inputG;
	}

	@Override
	public Object getOutputG(int id) {
		// TODO Auto-generated method stub
		return outputG;
	}

	@Override
	public Object cloneWithTiedParams() {
		// TODO Auto-generated method stub
		IdentityLayer clone = new IdentityLayer(length, linkId);
		return clone;
	}

}
