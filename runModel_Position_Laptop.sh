#!/usr/bin/env bash

#PBS -q APPLI
#PBS -o /home/s1620007/deepNN/outputs/deepNN_position.out
#PBS -e /home/s1620007/deepNN/outputs/deepNN_position.in
#PBS -N deepNN_position
#PBS -j oe

setenv PATH ${PBS_O_PATH}

root="./bin/classes"

java -Xmx8g -cp $root model.EntityMemNN_Position_1_Main \
-embeddingLength 300 \
-roundNum 300 \
-classNum 3 \
-accuracyThreshold 0.5 \
-trainFile $root/data/semeval14/Laptops_Train.xml.seg \
-testFile $root/data/semeval14/Laptops_Test_Gold.xml.seg \
-embeddingFile $root/vec/glove.42B.300d.txt \
-clippingThreshold 100 \
-randomizeBase 0.01 \
-modelFile ./results/laptop_p/model \
-attentionCellRandomBase 0.01 \
-attentionCellInnerORlinear linear \
-entityTransferRandomBase 0.01 \
-probThreshold 0.001 \
-numberOfHops 9 \
-learningRate 0.01 \
-isContainTargetInContext true \
-isNormLookup false \
-entityTransIdentityORlinear linear \
-trainedModel $root/trainedModel/Restaurant-hop9-model1--12.model \

