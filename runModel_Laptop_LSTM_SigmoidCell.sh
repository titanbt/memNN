#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/deepNN_java/outputs/deepNN_Lap_LSTM.out
#PBS -e /home/s1620007/deepNN_java/outputs/deepNN_Lap_LSTM.in
#PBS -N deepNN_Lap_LSTM
#PBS -j oe

cd /home/s1620007/deepNN_java

setenv PATH ${PBS_O_PATH}

root="./target/classes"

java -Xmx8g -cp $root model.LSTM_EntityMemNNMain_SigmoidCell \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.72 \
-trainFile $root/data/semeval14/Laptops_Train.xml.seg \
-testFile $root/data/semeval14/Laptops_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.01 \
-attentionCellRandomBase 0.01 \
-entityTransferRandomBase 0.01 \
-probThreshold 0.05 \
-numberOfHops 7 \
-learningRate 0.001 \
-batchSize 500 \
-randomSeed 2001 \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \
-modelFile ./results/lap_lstm_sigmoidcell/model \


