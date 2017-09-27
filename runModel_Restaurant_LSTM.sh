#!/usr/bin/env bash

#PBS -q SINGLE
#PBS -o /home/s1620007/deepNN_java/outputs/deepNN_Res_LSTM_8hop.out
#PBS -e /home/s1620007/deepNN_java/outputs/deepNN_Res_LSTM_8hop.in
#PBS -N 8hop_deepNN_Res_LSTM
#PBS -j oe

cd /home/s1620007/deepNN_java

setenv PATH ${PBS_O_PATH}

root="./target/classes"

java -Xmx8g -cp $root model.LSTM_EntityMemNNMain \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.8 \
-trainFile $root/data/semeval14/Restaurants_Train.xml.seg \
-testFile $root/data/semeval14/Restaurants_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.001 \
-modelFile ./results/res_3/model \
-attentionCellRandomBase 0.001 \
-attentionCellInnerORlinear linear \
-entityTransferRandomBase 0.01 \
-probThreshold 0.05 \
-numberOfHops 8 \
-learningRate 0.01 \
-isContainTargetInContext false \
-isNormLookup false \
-entityTransIdentityORlinear linear \
-batchSize 500 \
-randomSeed 2002 \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \


