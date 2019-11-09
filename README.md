# Co-reference resolution system based on Machine Learning

## Introduction
This was originally a class project written by Anneswa Ghosh and Myself (Sriram).

After exploring various options that were available, we chose to implement the simplest and straight-forward machine learning approach based on the below paper.

[A Machine Learning Approach to Coreference Resolution of Noun Phrases" paper by Soon, NG, et All.] (https://dl.acm.org/citation.cfm?id=972602)

##INPUT FILE SPECIFICATIONS:
Input files are present in data/train folder, they end with ".input" extension

Each document is marked up with two types of information:
	(1) sentence boundaries,and 
	(2) the initial reference for every coreference cluster.

Each sentence is surrounded by the tags<S ID=“#”>and</S>, where ID indicatesthe sentence number (#).

Each initial reference is surrounded by the tags<COREF ID=“X#”>and</COREF>,where the ID is a unique identifier (X#) for the reference. The initial reference is the earliest mention of the cluster concept in the document. One can assume that there will be at least one additional reference belonging to each cluster, that all additional references will occur after the initial reference, and that each reference will be a pronoun, noun, or nounphrase. For this project we didn't need to find possessive references (e.g., “his” or “Susan’s”).

As an example, a short story might look like this:

<S ID=“0”><COREF ID=“X0”>Susan Mills</COREF>bought<COREF ID=“X1”>ahome</COREF>in Utah.</S>
<S ID=“1”>A nice feature is that the 2-story house has a big yard for<COREF ID=“X2”>herdog</COREF>.</S>
<S ID=“2”>The German Shepherd weighs 100 lbs and is very active.</S><S ID=“3”>Both Sue and the dog love the new house!</S>

This story has 4 sentences and 3 coreference clusters. The initial reference for the first cluster is “Susan Mills”, the initial reference for the second cluster is “a home”, and the initial reference for the third cluster is “her dog”.

##ANSWER KEY (GOLD STANDARD) FILE:
Answer key files are for the development, they contain the gold standard coreference clusters for each document. We use the answer key files to better understand the nature of the task, to evaluate the performance of your system on thedevelopment set stories, and to train a machine learning system. 

Each coreference cluster in the answer key files begins with the initial reference (as marked up in the document) followed by each coreferent phrase in the document on a separate line. Each coreferent phrase is indicated by the sentence number where it occurs, its maximalphrase, and its minimal phrase. The maximal phrase is the full text span for the coreferent phrase, and the minimal phrase is the shortest acceptable text span for the coreferent phrase, which is usually the head noun or a named entity span. The specific format is shown below:

INITIAL REFERENCE
{SENTENCEID} {MAXIMALPHRASE} {MINIMALPHRASE}
{SENTENCEID} {MAXIMALPHRASE} {MINIMALPHRASE}
etc.

Each cluster’s information is separated by a blank line. For the sample document shownearlier, the answer key file looks like this:

<COREF ID=“X0”>Susan Mills</COREF>
{3} {Sue} {Sue}

<COREF ID=“X1”>a home</COREF>
{1} {the 2-story house} {house}
{3} {the new house} {house}

<COREF ID=“X2”>her dog</COREF>
{2} {The German Shepherd} {German Shepherd}
{3} {the dog} {dog}

Coreference system is scored based on its ability to find the correct coreferent phrases for each cluster, and not to produce spurious resolutions. A system’s response phrase is considered correct if the phrase is in the correct sentence, includes the minimal span,and does not exceed the maximal span.

For example, consider the coreference cluster for “a home”. Any of the following phrases from Sentence 1 is considered correct: “the 2-story house”, “2-story house”, and “house”. But these phrases are considered incorrect: “that the 2-story house” (exceeds maximalspan), “house has” (exceeds maximal span), and “the house” (because the phrase is not acontiguous span in the document).

OUTPUT (RESPONSE) FILE SPECIFICATIONS:
-------------------------------------
Coreference system should produce a response file for each document that lists the coreference phrases found for each cluster. The format is similar to the answer key files,except that it should produce just a single phrase for each answer (whereas the answer keyhas maximal and minimal spans). The specific format is shown below:

INITIAL REFERENCE
{SENTENCEID} {ANSWERPHRASE}
{SENTENCEID} {ANSWERPHRASE}
etc.

The information for each cluster is separated by a blank line. For example, a response file for the previous story looks like this:

<COREF ID=“X0”>Susan Mills</COREF>
{1} {A nice feature}
{3} {Sue}

<COREF ID=“X1”>a home</COREF>
{1} {house has}

<COREF ID=“X2”>her dog</COREF>
{3} {dog}

If the system produced this response file, two of the answers are considered correct(“Sue” and “dog”) and two of the answers are considered incorrect (“A nice feature”and “house has”). The system’s recall would be 2/5 because it found 2 of the 5 coreferent phrases in the answer key, and its precision would be 2/4 because 2 of the 4 answers that itgenerated are correct.

We list the mentions in the order that they occur in the document, with the earliest mention first. This is especially important when there are multiple referents in the same sentence.

For example, consider the sentence: John Smith is a singer and John has wonmany awards.
Be sure to list “John Smith” in your list of referents before “John”.

I/O SPECIFICATIONS:
-------------------
Coreference resolver accepts two arguments as input:
1. A ListFile that specifies the full pathname for the input les to be processed, one per line. Each input file is named \StoryID.input". For example, a ListFile might look like this:

/home/yourname/project/coref/data/1.input
/home/yourname/project/coref/data/52.input
/home/yourname/project/coref/data/33.input

2. A directory name (string), where the program's output (response files) will be written.

For example, when using python 3:

python3 coref <ListFile> <ResponseDir>

python3 coref test1.lisfile /home/yourname/project/coref/output/

For each input file, coreference resolver will produce a new file with the same prefix but the extension .response instead of .input.
For example, given 52.input, your program should generate an output file named 52.response. All of the response files should be put in the directory specified on the command line.

Scoring:
---------
The performance of the coreference resolver is based on its F score, which is a harmonic mean of Recall and Precision.
These scores are defined as:

Recall (R): the number of correct references identied by your system divided by the total number of references in the answer key.

Precision (P): the number of correct references identied by your system divided by the total number of references produced by your system.

F Score: F(R,P) = (2 x P x R)/(P + R)

This formula measures the balance between recall and precision (it is the harmonic mean).
The final performance of each system will be based on its F Score.

Mandatorily Required Packages:
------------------------------
python3.6
apt install python3-pip
pip3 install numpy nltk spacy
pip3 install scikit-learn
pip3 install prettytable --user
python3 -m spacy download en_core_web_sm

Inside python3.6 prompt
>>> import nltk
>>> nltk.download('wordnet')

Training From the Scratch:
-----------------------------
python3.6 ./feature_extract.py
cd models/Logistic_Regression/
python3.6 ./lr_train.py
cd ../../

Using Coref File to Test:
--------------------------
python3.6 ./coref.py list_file.txt scoring-program/responses/
