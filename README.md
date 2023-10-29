
# CS 769 Project

This repository holds the code for our group's main project for CS 769. It is a Bert model that trains on the go-emotions dataset that we have added a connection from the output of the first hidden layer to the classification layer for, therefore making it more syntax aware than a normal bert model.

To train the model on the origional go-emotions dataset, run `python main.py`. To train it on the "ekman" dataset, run `python main.py --taxonomy ekman`, and to run it on the "group" dataset, run `python main.py --taxonomy group`.


