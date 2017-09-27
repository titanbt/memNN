package model;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

import dataprepare.Data;
import dataprepare.Funcs;
import nn.libs.*;
import nn.libs.combinedLayer.*;
import evaluationMetric.Metric;
import nn.SentenceEntityMemNN;

// for laptop, the following parameters always work well

//-embeddingLength		300
//-roundNum		300
//-clippingThreshold		100
//-randomizeBase		0.01
//-dumpResultFile		tmpresults-laptop/log-no-use
//-attentionCellRandomBase		0.01
//-attentionCellInnerORlinear		linear
//-entityTransferRandomBase		0.01
//-numberOfHops		9
//-learningRate		0.01
//-isContainTargetInContext		true
//-isNormLookup		false
//-entityTransIdentityORlinear		linear
//-classNum		3
//-testFile		semeval14/Laptops_Test_Gold.xml.seg
//-embeddingFile		glove/glove.42B.300d.txt
//-trainFile		semeval14/Laptops_Train.xml.seg


// for restaurant, we can also use the previous params. we can also try the following setting
//==== begin configuration ====
//-embeddingLength		300
//-roundNum		300
//-clippingThreshold		100
//-randomizeBase		0.003
//-dumpResultFile		tmpresults-restaurant/log-no-use
//-attentionCellRandomBase		0.003
//-attentionCellInnerORlinear		linear
//-entityTransferRandomBase		0.01
//-numberOfHops		8
//-learningRate		0.01
//-isContainTargetInContext		false
//-isNormLookup		false
//-entityTransIdentityORlinear		linear
//-classNum		3
//-testFile		semeval14/Restaurants_Test_Gold.xml.seg
//-embeddingFile		glove/glove.42B.300d.txt
//-trainFile		semeval14/Restaurants_Train.xml.seg
//==== end configuration ====
public class EntityMemNNMain {
	
	LookupLayer seedLookup;
	ConnectLinearTanh attentionCellSeed;
	LinearLayer entityTranformSeed;
	LinearLayer linearForSoftmax;
	SoftmaxLayer softmax;
	HashMap<String, Integer> wordVocab = null;
	
	HashMap<Integer, String> reverseWordVocab = null;
	
	public EntityMemNNMain(String modelFile) throws Exception
	{
		TmpClass_ tmpClass = loadModel(modelFile);
		printArgs(tmpClass._argsMap);
		
		argsMap = tmpClass._argsMap;
		
		seedLookup = tmpClass._seedLookup;
		attentionCellSeed = tmpClass._attentionCellSeed;
		entityTranformSeed = tmpClass._entityTranformSeed;
		linearForSoftmax = tmpClass._linearForSoftmax;
		softmax = tmpClass._softmax;
		
		// do not forget to link!
		linearForSoftmax.link(softmax);
		
		wordVocab = tmpClass._wordVocab;
		reverseWordVocab = new HashMap<Integer, String>();
		for(String key: wordVocab.keySet())
		{
			reverseWordVocab.put(wordVocab.get(key), key);
		}
	}
	
	public EntityMemNNMain() throws Exception
	{
		Random rnd = new Random();
		int seed = Integer.parseInt(argsMap.get("-randomSeed"));
		rnd.setSeed(seed);

		int classNum = Integer.parseInt(argsMap.get("-classNum"));
		String trainFile = argsMap.get("-trainFile");
		String testFile  = argsMap.get("-testFile");
		int embeddingLength = Integer.parseInt(argsMap.get("-embeddingLength"));
		String embeddingFile = argsMap.get("-embeddingFile");
		double randomizeBase = Double.parseDouble(argsMap.get("-randomizeBase"));
		double attentionCellRandomBase = Double.parseDouble(argsMap.get("-attentionCellRandomBase"));
		double entityTransferRandomBase = Double.parseDouble(argsMap.get("-entityTransferRandomBase"));

		HashSet<String> wordSet = new HashSet<String>();
		loadData(trainFile, testFile, wordSet);		
		
		wordVocab = new HashMap<String, Integer>();
		double[][] table = Funcs.loadEmbeddingFile(embeddingFile, embeddingLength, "utf8", 
				false, wordVocab, wordSet);
		
		seedLookup = new LookupLayer(embeddingLength, wordVocab.size(), 1);
		seedLookup.setEmbeddings(table);


		linearForSoftmax = new LinearLayer(embeddingLength, classNum);
		linearForSoftmax.randomize(rnd, -1.0 * randomizeBase, randomizeBase);
		
		softmax = new SoftmaxLayer(classNum);

		linearForSoftmax.link(softmax);
		
		attentionCellSeed = new ConnectLinearTanh(embeddingLength, embeddingLength, 1);
		attentionCellSeed.randomize(rnd, -1.0 * attentionCellRandomBase, attentionCellRandomBase);
		
		entityTranformSeed = new LinearLayer(embeddingLength, embeddingLength);
		entityTranformSeed.randomize(rnd, -1.0 * entityTransferRandomBase, entityTransferRandomBase);
	}
	
	List<Data> trainDataList;
	
	public void loadData(
			String trainFile,
			String testFile,
			HashSet<String> wordSet)
	{
		System.out.println("================ start loading corpus ==============");
		
		trainDataList =  Funcs.loadCorpus(trainFile, "utf8");
		List<Data> testDataList = Funcs.loadCorpus(testFile, "utf8");

		List<Data> allList = new ArrayList<Data>();
		allList.addAll(trainDataList);
		allList.addAll(testDataList);
		
		for(Data data: allList)
		{
			String[] words = data.text.split(" ");
			for(String word: words)
			{
				if(word.equals("$t$"))
					continue;
				
				wordSet.add(word);
			}
			
			String[] targets = data.target.split(" ");
			for(String target: targets)
			{
				wordSet.add(target);
			}
		}
			
		System.out.println("training size: " + trainDataList.size());
		System.out.println("testDataList size: " + testDataList.size());
		System.out.println("wordSet.size: " + wordSet.size());
		
		System.out.println("================ finsh loading corpus ==============");
	}
	
	public String train(int round) throws Exception
	{
		int numberOfHops = Integer.parseInt(argsMap.get("-numberOfHops"));
		String modelFile = argsMap.get("-modelFile");
		double learningRate = Double.parseDouble(argsMap.get("-learningRate"));
		double probThreshold = Double.parseDouble(argsMap.get("-probThreshold"));
        int batchSize = Integer.parseInt(argsMap.get("-batchSize"));
		
		double lossV = 0.0;
		int lossC = 0;
		System.out.println("============== running round: " + round + " ===============");
		Collections.shuffle(trainDataList, new Random());
		System.out.println("Finish shuffling training data.");
		
		for(int idxData = 0; idxData < trainDataList.size(); idxData++)
		{
			Data data = trainDataList.get(idxData);
			
			String text = data.text;
			text = text.replaceAll("\\$t\\$", "");
			
			String[] words = text.split(" ");
			
			int[] wordIds = Funcs.fillSentence(words, wordVocab);
			
			// target word vec
			double[] targetVec = new double[seedLookup.embeddingLength];
			String[] targetWords = data.target.split(" ");
			int[] targetIds = Funcs.fillSentence(targetWords, wordVocab);
			
			if(targetIds.length == 0 || wordIds.length == 0)
			{
//				System.err.println("targetIds.length == 0 || forwWordIds.length == 0");
				continue;
			}
			
			for(int id: targetIds)
			{
				double[] xVec = seedLookup.table[id];
				for(int i = 0; i < xVec.length; i++)
				{
					targetVec[i] += xVec[i];
				}
			}
			
			for(int i = 0; i < targetVec.length; i++)
			{
				targetVec[i] = targetVec[i] / targetIds.length;
			}
			
			SentenceEntityMemNN memNN = new SentenceEntityMemNN(
					wordIds,
					seedLookup,
					attentionCellSeed,
					entityTranformSeed,
					targetVec,
					numberOfHops);
			
			// link. important.
			memNN.link(linearForSoftmax);
			
			memNN.forward();
			linearForSoftmax.forward();
			softmax.forward();
			
			// set cross-entropy error 
			int goldPol = data.goldPol;
			
			for(int k = 0; k < softmax.outputG.length; k++)
				softmax.outputG[k] = 0.0;
			
			if(softmax.output[goldPol] < probThreshold)
			{
				softmax.outputG[goldPol] =  1.0 / probThreshold;
				lossV += -Math.log(probThreshold);
			}
			else
			{
				softmax.outputG[goldPol] = 1.0 / softmax.output[goldPol];
				lossV += -Math.log(softmax.output[goldPol]);
			}
			lossC += 1;
			lossC += 1;
			
			// backward
			softmax.backward();
			linearForSoftmax.backward();
			memNN.backward();
			
			// update
			linearForSoftmax.update(learningRate);
			memNN.update(learningRate);
			
//			linearForSoftmax.updateAdaGrad(learningRate, 300);
//			memNN.updateAdaGrad(learningRate, 300);
			
			// clearGrad
			memNN.clearGrad();
			linearForSoftmax.clearGrad();
			softmax.clearGrad();

			if(idxData % batchSize == 0)
			{
				System.out.println("running idxData = " + idxData + "/" + trainDataList.size() + "\t "
						+ "lossV/lossC = " + String.format("%.4f", lossV) + "/" + lossC + "\t"
						+ " = " + String.format("%.4f", lossV/lossC)
						+ "\t" + new Date().toLocaleString());
			}
		}
		System.out.println("============= finish training round: " + round + " ==============");
		
		dumpToFile(modelFile + "-" + round + ".model");
		return modelFile + "-" + round + ".model";
	}

	public double predict(String modelFile) throws Exception
	{
		String testFile = argsMap.get("-testFile");
		List<Data> testDataList = Funcs.loadCorpus(testFile, "utf8");
		
//		String dumpResultsFile = modelFile + ".attentionResults";
//		PrintWriter writer = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
//		          new FileOutputStream(dumpResultsFile), "utf8")));
		
		int numberOfHops = Integer.parseInt(argsMap.get("-numberOfHops"));
		double accuracyThreshold = Double.parseDouble(argsMap.get("-accuracyThreshold"));
		
		System.out.println("===========start to predict===============");
		
		List<Integer> goldList = new ArrayList<Integer>();
		List<Integer> predList = new ArrayList<Integer>();
		
		for(int idxData = 0; idxData < testDataList.size(); idxData++)
		{
			Data data = testDataList.get(idxData);
			
			String text = data.text;
			text = text.replaceAll("$t$", "");
			
			String[] words = text.split(" ");
			int[] wordIds = Funcs.fillSentence(words, wordVocab);
			
			// target word vec
			double[] targetVec = new double[seedLookup.embeddingLength];
			String[] targetWords = data.target.split(" ");
			int[] targetIds = Funcs.fillSentence(targetWords, wordVocab);
			
			if(targetIds.length == 0 || wordIds.length == 0)
			{
//				System.err.println("targetIds.length == 0 || forwWordIds.length == 0");
				continue;
			}
			
			for(int id: targetIds)
			{
				double[] xVec = seedLookup.table[id];
				for(int i = 0; i < xVec.length; i++)
				{
					targetVec[i] += xVec[i];
				}
			}
			
			for(int i = 0; i < targetVec.length; i++)
			{
				targetVec[i] = targetVec[i] / targetIds.length;
			}
			
			SentenceEntityMemNN memNN = new SentenceEntityMemNN(
					wordIds,
					seedLookup,
					attentionCellSeed,
					entityTranformSeed,
					targetVec,
					numberOfHops);
			
			// link. important.
			memNN.link(linearForSoftmax);
			
			memNN.forward();
			linearForSoftmax.forward();
			softmax.forward();
			
			int predClass = -1;
			double maxPredProb = -1.0;
			for(int ii = 0; ii < softmax.length; ii++)
			{
				if(softmax.output[ii] > maxPredProb)
				{
					maxPredProb = softmax.output[ii];
					predClass = ii;
				}
			}
			
			predList.add(predClass);
			goldList.add(data.goldPol);
			
//			dumpAttentionWeights(
//					memNN, 
//					reverseWordVocab, 
//					data.text, 
//					data.target, 
//					wordIds, 
//					targetIds,
//					predClass,
//					data.goldPol,
//					writer);
		}
//		writer.close();
		double accuracy = Metric.calcMetric(goldList, predList);
		
		if(accuracy < accuracyThreshold)
		{
			File tmpModel = new File(modelFile);
			tmpModel.delete();
		}

		System.out.println("============== finish predicting =================");
		return accuracy;
	}

	static HashMap<String, String> argsMap = null;
	
	static void printArgs(HashMap<String, String> _map)
	{
		System.out.println("==== begin configuration ====");
		for(String key: _map.keySet())
		{
			System.out.println(key + "\t\t" + _map.get(key));
		}
		System.out.println("==== end configuration ====");
	}
	
	public void dumpToFile(String dumpModelFile) throws Exception
	{
		PrintWriter bw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(
		          new FileOutputStream(dumpModelFile), "utf8")));
		
		dumpToStream(bw);
		
		bw.close();
	}
	
	public void dumpToStream(Writer bw)
				throws Exception
	{
		bw.write(argsMap.size() + "\n");
		for(String key: argsMap.keySet())
		{
			bw.write(key + "\t\t" + argsMap.get(key) + "\n");
		}
		
		bw.write(wordVocab.size() + "\n");
		for(String key: wordVocab.keySet())
		{
			bw.write(key + "\t\t" + wordVocab.get(key) + "\n");
		}
		
		seedLookup.dumpToStream(bw);
		entityTranformSeed.dumpToStream(bw);
		linearForSoftmax.dumpToStream(bw);
		softmax.dumpToStream(bw);
		attentionCellSeed.dumpToStream(bw);
	}
	
	class TmpClass_{
		HashMap<String, Integer> _wordVocab;
		LookupLayer _seedLookup;
		HashMap<String, String> _argsMap;
		LinearLayer _linearForSoftmax;
		SoftmaxLayer _softmax;
		ConnectLinearTanh _attentionCellSeed;
		LinearLayer _entityTranformSeed;
		
		public TmpClass_(BufferedReader br) throws Exception
		{
			_argsMap = new HashMap<String, String>();
			int argLineNum = Integer.parseInt(br.readLine());
			for(int i = 0; i < argLineNum; i++)
			{
				String argLine = br.readLine();
				String[] splits = argLine.split("\t\t");
				if(splits.length != 2)
				{
					throw new Exception("format error, argsmap.splits[i].length != 2");
				}
				_argsMap.put(splits[0], splits[1]);
			}
			
			int userVocabSize = Integer.parseInt(br.readLine());
			_wordVocab = new HashMap<String, Integer>();
			
			for(int i = 0; i < userVocabSize; i++)
			{
				String wordKVLine = br.readLine();
				String[] splits = wordKVLine.split("\t\t");
				if(splits.length != 2)
				{
					throw new Exception("format error, _wordVocab.splits[i].length != 2");
				}
				_wordVocab.put(splits[0], Integer.parseInt(splits[1]));
			}
			
			_seedLookup = LookupLayer.loadFromStream(br);
			_entityTranformSeed = LinearLayer.loadFromStream(br);
			_linearForSoftmax = LinearLayer.loadFromStream(br);
			_softmax = SoftmaxLayer.loadFromStream(br);
			_attentionCellSeed = ConnectLinearTanh.loadFromStream(br);
		}
	}
	
	// fuck java. it does not support passing reference
	public TmpClass_ loadModel(String modelFile) throws Exception
	{
		BufferedReader br = new BufferedReader(new InputStreamReader(
				new FileInputStream(modelFile) , "utf8"));
		
		TmpClass_ tmpClass = new TmpClass_(br);
		br.close();
		
		return tmpClass;
	}
	
	public void dumpAttentionWeights(SentenceEntityMemNN memNN,
			HashMap<Integer, String> _reverseWordVocabMap,
			String orgText,
			String targetText,
			int[] wordIds,
			int[] targetIds,
			int predClass,
			int goldClass,
			Writer wr
			) throws IOException
	{
		wr.write("orgText: " + orgText + "\n");
		wr.write("targetText: " + targetText + "\n");

		for(int i = 0; i < targetIds.length; i++)
		{
			wr.write(_reverseWordVocabMap.get(targetIds[i]) + "\t");
		}
		wr.write("\n");
		
		for(int i = 0; i < wordIds.length; i++)
		{
			wr.write(_reverseWordVocabMap.get(wordIds[i]) + "\t");
		}
		wr.write("\n");
		
		wr.write("predClass: " + predClass + "\n");
		wr.write("goldClass: " + goldClass + "\n");
		
		for(AttentionLayer layer: memNN.attentionLayers)
		{
			for(int i = 0; i < layer.attentionSoftmax.output.length; i++)
			{
				wr.write(String.format("%.4f", layer.attentionSoftmax.output[i]) + "\t");
			}
			wr.write("\n");
		}
		
		wr.write("\n");
	}
	
	public static void main(String[] args) throws Exception
	{
		argsMap = Funcs.parseArgs(args);
		printArgs(argsMap);

		EntityMemNNMain main = new EntityMemNNMain();
		int roundNum = Integer.parseInt(argsMap.get("-roundNum"));
		double accuracy = 0.0;
		double bestAcc = 0.0;
		String bestModelFile = "";

		for(int round = 1; round <= roundNum; round++)
		{
			String modelFile = main.train(round);
			accuracy = main.predict(modelFile);
			if(bestAcc < accuracy) {
				bestAcc = accuracy;
				bestModelFile = modelFile;
			}
		}
		System.out.println("The best accuracy:" + String.format("%.4f", bestAcc) + " - Model file:" + bestModelFile);
		
//		String modelFile = "model/Restaurant-hop9-model1--12.model";
//		EntityMemNNMain main = new EntityMemNNMain(modelFile);
//		main.predict(modelFile);
	}
}
