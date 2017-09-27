#!/usr/bin/env bash

#PBS -q SINGLE
#PBS -o /home/s1620007/deepNN_java/outputs/deepNN_Lap_LSTM_7hop.out
#PBS -e /home/s1620007/deepNN_java/outputs/deepNN_Lap_LSTM_7hop.in
#PBS -N 7hop_deepNN_Lap_LSTM
#PBS -j oe

cd /home/s1620007/deepNN_java

setenv PATH ${PBS_O_PATH}

root="./target/classes"

java -Xmx8g -cp $root model.LSTM_InnerProduct_EntityMemNNMain \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.72 \
-trainFile $root/data/semeval14/Laptops_Train.xml.seg \
-testFile $root/data/semeval14/Laptops_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.01 \
-modelFile ./results/lap_lstm_hop1/model \
-attentionCellRandomBase 0.01 \
-attentionCellInnerORlinear linear \
-entityTransferRandomBase 0.01 \
-probThreshold 0.05 \
-numberOfHops 10 \
-learningRate 0.001 \
-isContainTargetInContext true \
-isNormLookup false \
-entityTransIdentityORlinear linear \
-batchSize 500 \
-randomSeed 2001 \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \


