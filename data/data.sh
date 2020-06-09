#!/usr/bin/env bash
source ~/.bash_profile

#wget http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/Musical_Instruments.json.gz
#wget http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_Musical_Instruments.json.gz

#gunzip -c ./meta_Musical_Instruments.json.gz > ./meta_Musical_Instruments.json
#gunzip -c ./Musical_Instruments.json.gz > ./Musical_Instruments.json

python2.7 preprocess.py