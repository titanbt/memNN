package analysis;

import java.util.ArrayList;
import java.util.HashMap;

import dataprepare.IO;

public class CompareBetween2 {

	static void run(String file1, String file2)
	{
		ArrayList<String> lists1 = IO.readFile(file1, "utf8");
		ArrayList<String> lists2 = IO.readFile(file2, "utf8");
		
		for(int i = 0; i < lists1.size(); i++)
		{
			String line1 = lists1.get(i);
			String line2 = lists2.get(i);
			
			String[] splits1 = line1.split("\t");
			String[] splits2 = line2.split("\t");
			
//			System.out.println(splits.length);
//			for(int k = 0; k < splits.length; k++)
//				System.out.println(k + " " + splits[k]);
			String key1 = splits1[1] + " " + splits1[2];
			String key2 = splits2[1] + " " + splits2[2];
			
			if(!key1.equals("gold=2 pred=2") && key2.equals("gold=2 pred=2"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
			
			if(!key1.equals("gold=1 pred=1") && key2.equals("gold=1 pred=1"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
			
			if(!key1.equals("gold=0 pred=0") && key2.equals("gold=0 pred=0"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
		}
		
	}
	
	static void run2(String file1, String file2, String file3)
	{
		ArrayList<String> lists1 = IO.readFile(file1, "utf8");
		ArrayList<String> lists2 = IO.readFile(file2, "utf8");
		ArrayList<String> lists3 = IO.readFile(file3, "utf8");
		
		for(int i = 0; i < lists1.size(); i++)
		{
			String line1 = lists1.get(i);
			String line2 = lists2.get(i);
			String line3 = lists3.get(i);
			
			String[] splits1 = line1.split("\t");
			String[] splits2 = line2.split("\t");
			String[] splits3 = line3.split("\t");
			
//			System.out.println(splits.length);
//			for(int k = 0; k < splits.length; k++)
//				System.out.println(k + " " + splits[k]);
			String key1 = splits1[1] + " " + splits1[2];
			String key2 = splits2[1] + " " + splits2[2];
			String key3 = splits3[1] + " " + splits3[2];
			
			if(!key1.equals("gold=2 pred=2") 
					&& key2.equals("gold=2 pred=2")
					&& key3.equals("gold=2 pred=2"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
			
			if(!key1.equals("gold=1 pred=1") 
					&& key2.equals("gold=1 pred=1")
					&& key3.equals("gold=1 pred=1"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
			
			if(!key1.equals("gold=0 pred=0") 
					&& key2.equals("gold=0 pred=0")
					&& key3.equals("gold=0 pred=0"))
			{
				System.out.println(line1);
				System.out.println(line2);
				System.out.println();
			}
		}
		
	}
	
	public static void main(String[] args) {
		run("analysis/lstm-combine.txt", "analysis/bi-lstm-combine.txt");
		System.out.println("===========================");
		run2("analysis/lstm-combine.txt", "analysis/bi-lstm-combine.txt", 
				"analysis/bi-lstm-target-combine.txt");
	}

}
