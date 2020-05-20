#!/bin/bash

list_train="./data/list_ModelNet10_train.txt"
list_test="./data/list_ModelNet10_test.txt"
label_train="./data/label_ModelNet10_train.txt"
label_test="./data/label_ModelNet10_test.txt"

shaperep_train="./data/train.shaperep"
shaperep_test="./data/test.shaperep"

device="/gpu:0"
checkpoint="./checkpoint" # learned parameters are saved in this directory

ndim=64 # number of dimensions for latent feature space
# rm -rf $checkpoint
command="python -u LearnSAT.py $shaperep_train $list_train $label_train $shaperep_test $list_test $label_test $ndim $device $checkpoint"
echo $command
$command

exit
