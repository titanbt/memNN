package analysis;

import java.util.ArrayList;
import java.util.HashMap;

import dataprepare.IO;

public class ErrorTable {
	
	static void run(String file)
	{
		System.out.println("Running file: " + file);
		ArrayList<String> lists = IO.readFile(file, "utf8");
		
		HashMap<String, Integer> errorMap = new HashMap<String, Integer>();
		
		for(String line: lists)
		{
			String[] splits = line.split("\t");
//			System.out.println(splits.length);
//			for(int k = 0; k < splits.length; k++)
//				System.out.println(k + " " + splits[k]);
			String key = splits[1] + " " + splits[2];
			
			if(!errorMap.containsKey(key))
				errorMap.put(key, 0);
			
			errorMap.put(key, errorMap.get(key) + 1);
		}
		
		for(String key: errorMap.keySet())
			System.out.println(key + " " + errorMap.get(key)
			);
		
		System.out.println("========================");
	}
	
	public static void main(String[] args) {
		run("analysis/lstm-combine.txt");
		run("analysis/bi-lstm-combine.txt");
		run("analysis/bi-lstm-target-combine.txt");
		run("analysis/bi-lstm-attention-combine.txt");
	}

}
