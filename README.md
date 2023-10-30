
# CS 769 Project

This repository holds the code for our group's main project for CS 769.

To train the model on the origional go-emotions dataset, run `python main.py`. To train it on the "ekman" dataset, run `python main.py --taxonomy ekman`, and to run it on the "group" dataset, run `python main.py --taxonomy group`.

You can also choose whether it uses the origional go-emotions model or our modified go-emotions model via the `--model_type` flag. By default, the modofied go-emotions model is used.
