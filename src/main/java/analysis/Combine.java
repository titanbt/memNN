package analysis;

import java.util.ArrayList;
import java.util.List;

import dataprepare.IO;

public class Combine {

	public void run(String gFile, String pFile, String srcFile, String oFile)
	{
		ArrayList g = IO.readFile(gFile, "utf8");
		ArrayList p = IO.readFile(pFile, "utf8"); 
		ArrayList src = IO.readFile(srcFile, "utf8");
		
		if(g.size() != p.size() || g.size() * 3 != src.size())
			System.err.println("wrong");
			
		List<String> outList = new ArrayList<String>();
		
		for(int i = 0; i < g.size(); i++)
		{
			String oStr = 	"id=" + (i+1) + "\t" + 
							"gold=" + g.get(i) + "\t" +
							"pred=" + p.get(i) + "\t\t";
			
			oStr += src.get(i * 3 + 1) + "\t\t\t" + src.get(i * 3);
			
			outList.add(oStr);
		}
		IO.writeFile(oFile, outList, "utf8");
	}
	
	public static void main(String[] args) {
		Combine main = new Combine();
		main.run(	"analysis/bi-lstm-attention-gold.txt", 
					"analysis/bi-lstm-attention-pred.txt",
					"acl-14-short-data/test.raw",
					"analysis/bi-lstm-attention-combine.txt");
	}

}
