#!/usr/bin/env bash

#PBS -q SINGLE
#PBS -o /home/s1620007/deepNN_java/outputs/deepNN_Res_LSTM_pos.out
#PBS -e /home/s1620007/deepNN_java/outputs/deepNN_Res_LSTM_pos.in
#PBS -N deepNN_Res_LSTM_pos
#PBS -j oe

cd /home/s1620007/deepNN_java

setenv PATH ${PBS_O_PATH}

root="./target/classes"

java -Xmx8g -cp $root model.LSTM_EntityMemNNMain_Position \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.8 \
-trainFile $root/data/semeval14/Restaurants_Train.xml.seg \
-testFile $root/data/semeval14/Restaurants_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.003 \
-modelFile ./results/res_lstm_position/model \
-attentionCellRandomBase 0.003 \
-attentionCellInnerORlinear linear \
-entityTransferRandomBase 0.01 \
-probThreshold 0.05 \
-numberOfHops 10 \
-learningRate 0.01 \
-isContainTargetInContext false \
-isNormLookup false \
-entityTransIdentityORlinear linear \
-batchSize 500 \
-randomSeed 2002 \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \
-positionThreshold 3 \
-positionRandomBase 0.01 \


