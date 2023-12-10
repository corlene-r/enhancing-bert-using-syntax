import os
import warnings
from argparse import ArgumentParser
from nltk.parse.stanford import StanfordParser
from nltk.tree.tree import Tree

warnings.filterwarnings("ignore", category=DeprecationWarning)

java_path = "C:\\Program Files\\Java\\jdk-15\\bin"
os.environ['JAVAHOME'] = java_path

stanford_parser_dir = './stanford-parser-full-2020-11-17'
os.environ['STANFORD_PARSER'] = f'{stanford_parser_dir}/stanford-parser.jar'
os.environ['STANFORD_MODELS'] = f'{stanford_parser_dir}/stanford-parser-4.2.0-models.jar'

parser = StanfordParser(model_path="edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz", encoding='utf8', java_options="-Xmx1g -Xmx4g -Xmx8g -Xmx16g -Xmx32g")

def main():
    # create arg parser and read args

    arg_parser = ArgumentParser(prog="Stanford Parser")
    arg_parser.add_argument('input_file', type=str, help="The file to output the results to.")
    arg_parser.add_argument('output_file', type=str, help="The file to output the results to.")
    arg_parser.add_argument('-s', '--start_line', type=int, help="The line of the file to start at.")
    arg_parser.add_argument('-e', '--end_line', type=int, help="The line of the file to end at.")
    args = arg_parser.parse_args()

    start_line = args.start_line or 0

    # open up file and read lines
    lines = []
    with open(args.input_file, 'r', encoding='utf8') as file:
        # consume the first "start_line" lines of the file
        for _ in range(start_line):
            line = file.readline()
            if len(line) == 0: break

        # add (end_line - start_line) lines to "lines"
        for _ in range(start_line, args.end_line):
            line = file.readline()
            if len(line) == 0: break

            # escape characters
            line = line.strip()

            # do not add empty lines
            if len(line) == 0:
                continue

            # escape some characters of the line
            line = line \
                    .replace("\\" , "\\\\") \
                    .replace('\n', "\\n") \
                    .replace('\r', "\\r") \
                    .replace('"' , '\\"') \
                    .replace("'" , "\\'") \

            # add the line of text
            lines.append(line.split('\t')[0])

    # parse the lines into trees

    def parse(sent):
        try:
            return list(parser.raw_parse(sent))
        except RuntimeError:
            return Tree("ROOT", ["ERROR", "SENTENCE", "TOO", "LONG"])

    sentences = [parse(sent) for sent in lines]

    def paren(tree):
        """
        A helper method to recursively take a node in the parsed tree and turn it into a string.
        """
        label = f'"{tree.label()}"'
        out = [f'"{n}"' if isinstance(n, str) else paren(n) for n in tree]
        strs = ','.join(out)
        strs = ',' + strs if strs else ''
        return f"({label}{strs})"

    # turn the trees into strings
    str_trees = ''.join(f'{paren(sent[0])}\n' for sent in sentences)

    # write the trees
    with open(args.output_file, 'w+', encoding='utf8') as out:
        out.write(str_trees)

if __name__ == "__main__":
    main()
