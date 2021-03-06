#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/deepNN_java/outputs/deepNN_Lap.out
#PBS -e /home/s1620007/deepNN_java/outputs/deepNN_Lap.in
#PBS -N deepNN_Lap
#PBS -j oe

cd /home/s1620007/deepNN_java

setenv PATH ${PBS_O_PATH}

root="./target/classes"

java -Xmx8g -cp $root model.EntityMemNNMain \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.5 \
-trainFile $root/data/semeval14/Laptops_Train.xml.seg \
-testFile $root/data/semeval14/Laptops_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.01 \
-modelFile ./results/laptop/model \
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


