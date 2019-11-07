#Overall Description:
---------------------
Project is implemented basing on "A Machine Learning Approach to Coreference Resolution of Noun Phrases" paper by Soon, NG, et All.
It uses Machine Learning Model based on Mention Pairs.

#Mandatorily Required Packages:
------------------------------
python3.6
apt install python3-pip
pip3 install numpy nltk spacy
pip3 install scikit-learn
pip3  install prettytable --user
python3 -m spacy download en_core_web_sm

#Inside python3.6 prompt
>>> import nltk
>>> nltk.download('wordnet')

#Training From the Scratch:
-----------------------------
python3.6 ./feature_extract.py
cd models/Logistic_Regression/
python3.6 ./lr_train.py
cd ../../

#Using Coref File to Test:
--------------------------
python3.6 ./coref.py list_file.txt scoring-program/responses/
