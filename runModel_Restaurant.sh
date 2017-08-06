#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/deepNN/outputs/deepNN_Res.out
#PBS -e /home/s1620007/deepNN/outputs/deepNN_Res.in
#PBS -N deepNN_Res
#PBS -j oe

cd /home/s1620007/deepNN

setenv PATH ${PBS_O_PATH}

root="./bin/classes"

java -Xmx8g -cp $root model.EntityMemNNMain \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.5 \
-trainFile $root/data/semeval14/Restaurants_Train.xml.seg \
-testFile $root/data/semeval14/Restaurants_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.01 \
-modelFile ./results/restaurant/model \
-attentionCellRandomBase 0.01 \
-attentionCellInnerORlinear linear \
-entityTransferRandomBase 0.01 \
-probThreshold 0.05 \
-numberOfHops 9 \
-learningRate 0.01 \
-isContainTargetInContext true \
-isNormLookup false \
-entityTransIdentityORlinear linear \
-batchSize 500 \
-randomSeed 1189 \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \


