
# CS 769 Project: UW Madison FA 23
### Authors: Alexander Peseckis, Corlene Rhoades, Ojas Sethi

This repository holds the code for our group's main project for CS 769. Our project is titled 'Enhancing BERT Sentiment Classification Syntactic Information.'

To install the dependencies, run ```pip install -r requirements.txt``` from the main directory.

To train the model on the origional go-emotions dataset, run `python main.py`. To train it on the "ekman" dataset, run `python main.py --taxonomy ekman`, and to run it on the "group" dataset, run `python main.py --taxonomy group`.

You can also choose whether it uses the origional go-emotions model or our modified go-emotions model via the `--model_type` flag. By default, the modofied go-emotions model is used.

The syntax parser that we used was the Stanford Parser. It's not included in this repository because of file size constraints. However, the ```.jar``` file can be downloaded from this link: [Stanford Parser](https://nlp.stanford.edu/software/lex-parser.shtml#Download).
