The A5 Problem 2 script is contained in the file p2.py.

The script trains a BPE model on training data.
Saves the encoded output to a file. 
Saves a plot of relevant training data.
Encodes new words using the trained model, and saves that output to a file.

There are no parameters to set to run the model. 

A5-data.txt contains the relevant data to train the BPE model.
least_common_words.txt contains releveant data for encoding new words using the trained model.

Run:
    python3 p2.py

Dependencies:
    A data directory at '../data/'.
    'A5-data.txt' exists in data.
    'least_common_words.txt' exists in data.
