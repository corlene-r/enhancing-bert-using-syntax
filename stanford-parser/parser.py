import os
from nltk.parse.stanford import StanfordParser
import nltk
from nltk import tree

java_path = "/usr/bin/java"
os.environ['JAVAHOME'] = java_path

stanford_parser_dir = '/Users/ojassethi/Desktop/parser/stanford-parser-full-2020-11-17'
os.environ['STANFORD_PARSER'] = f'{stanford_parser_dir}/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = f'{stanford_parser_dir}/stanford-parser-4.2.0-models.jar'

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz")

def parse_sentences_from_tsv(input_file, output_file):
    with open(input_file, 'r') as file:
        sentences = file.read().split('\t')

    with open(output_file, 'w') as out:
        total = len(sentences)
        print(f"Starting parsing {total} sentences.")

        for i, sentence in enumerate(sentences, 1):
            parsed_trees = list(parser.raw_parse(sentence))
            for tree in parsed_trees:
                out.write(str(tree) + '\n\n')
            
            print(f"Parsed {i}/{total} sentences.")

    parser.java_options = '-mx1000m'

parse_sentences_from_tsv('test.tsv', 'output.txt')

print("Parsing completed.")