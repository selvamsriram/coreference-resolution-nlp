#!/bin/bash
apt-get update
apt-get install pip3.6
pip3.6 install numpy nltk spacy scikit-learn prettytable 
python3.6 -m spacy download en_core_web_sm
