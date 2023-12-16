import os
import copy
import json
import logging
from nltk.tree.tree import Tree
from difflib import get_close_matches

import torch
from torch.utils.data import TensorDataset

logger = logging.getLogger(__name__)

from stanford_parser.load_trees import load_trees_from

class InputExample(object):
    """ A single training/test example for simple sequence classification. """

    def __init__(self, guid, text_a, text_b, label):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

def convert_examples_to_features(
        args,
        examples,
        tokenizer,
        max_length,
        tree_file
):
    def tree_dist(location1, location2):
        # Distances between two nodes of a tree given the path to them
        i = 0
        while i < len(location1) and i < len(location2) and location1[i] == location2[i]:
            i+=1
        return len(location1) - i + len(location2) - i
    trees = load_trees_from(tree_file)
    processor = GoEmotionsProcessor(args)
    label_list_len = len(processor.get_labels())

    def convert_to_one_hot_label(label):
        one_hot_label = [0] * label_list_len
        for l in label:
            one_hot_label[l] = 1
        return one_hot_label

    labels = [convert_to_one_hot_label(example.label) for example in examples]

    batch_encoding = tokenizer.batch_encode_plus(
        [(example.text_a, example.text_b) for example in examples], max_length=max_length, pad_to_max_length=True
    )

    features = []
    for i, example in enumerate(examples):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}
        
        feature = InputFeatures(**inputs, label=labels[i])
        tokens =  tokenizer.tokenize(example.text_a)
        leaves = trees[i][0].leaves()
        tree_idxs = []
        for j, t in enumerate(tokens[:max_length]): 
            closest = get_close_matches(t, leaves, n=1)
            if closest: idxs_in_tree = [k for k in range(len(leaves)) if leaves[k] == closest[0]]
            else: idxs_in_tree = [-1]
            idx = min(idxs_in_tree, key=lambda x:abs(x-j))
            tree_idxs.append(idx)
        
        tree_idxs = [idx if idx == -1 else trees[i][0].leaf_treeposition(idx) for idx in tree_idxs]

        attention_mask = torch.zeros(max_length, max_length, dtype=torch.long)
        for j in range(len(tokens[:max_length])):
            for i in range(len(tokens[:max_length])):
                if abs(i - j) < args.adjacency_threshhold:
                    attention_mask[i, j] = 1
                if tree_idxs[i] != -1 and tree_idxs[j] != -1 and tree_dist(tree_idxs[i], tree_idxs[j]) <= args.tree_threshhold:
                    attention_mask[i, j] = 1
                    attention_mask[j, i] = 1
        feature.attention_mask = attention_mask
        features.append(feature)

    for i, example in enumerate(examples[:10]):
        logger.info("*** Example ***")
        logger.info("guid: {}".format(example.guid))
        logger.info("sentence: {}".format(example.text_a))
        logger.info("tokens: {}".format(" ".join([str(x) for x in tokenizer.tokenize(example.text_a)])))
        logger.info("input_ids: {}".format(" ".join([str(x) for x in features[i].input_ids])))
        logger.info("attention_mask: {}".format(" ".join([str(x) for x in features[i].attention_mask])))
        logger.info("token_type_ids: {}".format(" ".join([str(x) for x in features[i].token_type_ids])))
        logger.info("label: {}".format(" ".join([str(x) for x in features[i].label])))

    return features


class GoEmotionsProcessorWithTrees(object):
    """Processor for the GoEmotions data set """

    def __init__(self, args):
        self.args = args

    def get_labels(self):
        labels = []
        with open(os.path.join(self.args.data_dir, self.args.label_file), "r", encoding="utf-8") as f:
            for line in f:
                labels.append(line.rstrip())
        return labels

    @classmethod
    def _read_file(cls, input_file):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            return f.readlines()

    def _create_examples(self, lines, set_type):
        """ Creates examples for the train, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            line = line.strip()
            items = line.split("\t")
            text_a = items[0]
            label = list(map(int, items[1].split(",")))
            if i % 5000 == 0:
                logger.info(line)
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = self.args.train_file
        elif mode == 'dev':
            file_to_read = self.args.dev_file
        elif mode == 'test':
            file_to_read = self.args.test_file

        logger.info("LOOKING AT {}".format(os.path.join(self.args.data_dir, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.args.data_dir,
                                                                  file_to_read)), mode)


def load_and_cache_examples_with_trees(args, tokenizer, mode):
    processor = GoEmotionsProcessor(args)
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            str(args.task),
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_len),
            mode
        )
    )
    if False: #os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode == "train":
            examples = processor.get_examples("train")
            tree_file = args.train_tree_file
        elif mode == "dev":
            examples = processor.get_examples("dev")
            tree_file = args.dev_tree_file
        elif mode == "test":
            examples = processor.get_examples("test")
            tree_file = args.test_tree_file
        else:
            raise ValueError("For mode, only train, dev, test is available")
        features = convert_examples_to_features(
            args, examples, tokenizer, max_length=args.max_seq_len, tree_file=tree_file)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.stack([f.attention_mask for f in features])
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset
