#!/bin/bash

echo "starting a pos"
python bilstmTrain.py a data/pos/train a_model_pos pos data/pos/dev 
echo "starting b pos"
python bilstmTrain.py b data/pos/train b_model_pos pos data/pos/dev
echo "starting c pos"
python bilstmTrain.py c data/pos/train c_model_pos pos data/pos/dev
echo "starting d pos"
python bilstmTrain.py d data/pos/train d_model_pos pos data/pos/dev
echo "starting a ner"
python bilstmTrain.py a data/ner/train a_model_ner ner data/ner/dev 
echo "starting b ner"
python bilstmTrain.py b data/ner/train b_model_ner ner data/ner/dev 
echo "starting c ner"
python bilstmTrain.py c data/ner/train c_model_ner ner data/ner/dev 
echo "starting d ner"
python bilstmTrain.py d data/ner/train d_model_ner ner data/ner/dev 

