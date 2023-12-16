from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from data_loader_syntax_mask import (
    load_and_cache_examples,
    GoEmotionsProcessor
)
import argparse
import os
import json
from bert_multi_label_classifier import BertForMultiLabelClassification
import torch

GO_EMOTIONS = "goemotions"
GO_EMOTIONS_MODIFIED = "goemotions_modified"
GO_EMOTIONS_SYNTAX_FED = "goemotions_syntax_fed"
GO_EMOTIONS_SYNTAX_INTERLEAVED = "goemotions_syntax_interleaved"

def main(cli_args):
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("config", config_filename)) as f:
        args = argparse.Namespace(**json.load(f))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    args.taxonomy = cli_args.taxonomy
    args.model_type = cli_args.model_type

    processor = GoEmotionsProcessor(args)
    label_list = processor.get_labels()

    config = BertConfig.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        finetuning_task=args.task,
        id2label={str(i): label for i, label in enumerate(label_list)},
        label2id={label: i for i, label in enumerate(label_list)}
    )
    tokenizer = BertTokenizer.from_pretrained(
        args.tokenizer_name_or_path,
    )

    # Create the Model
    # if args.model_type == GO_EMOTIONS:
    #     model = BertForMultiLabelClassification.from_pretrained(
    #         args.model_name_or_path,
    #         config=config
    #     )
    # elif args.model_type == GO_EMOTIONS_MODIFIED:
    #     model = ModifiedBertForMultiLabelClassification.from_pretrained(
    #         args.model_name_or_path,
    #         config=config
    #     )
    # elif args.model_type == GO_EMOTIONS_SYNTAX_FED:
    #     model = SyntaxFedBertForMultiLabelClassification.from_pretrained(
    #         args.model_name_or_path,
    #         config=config
    #     )
    # elif args.model_type == GO_EMOTIONS_SYNTAX_INTERLEAVED:
    #     model = SyntaxInterleavedBertForMultiLabelClassification.from_pretrained(
    #         args.model_name_or_path,
    #         config=config
    #     )
    # else:
    #     raise NotImplementedError(f"{args.model_type} is an unknown model option")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    # model.to(args.device)

    # run_id = str(time.time())

    args.train_file       = "small.tsv"
    args.train_tree_file  = "data/original/small_trees.tsv"
    args.tree_threshhold  = 7
    args.adjacency_threshhold = 3
    # Load dataset
    print("hello2")
    train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
    print("hello3")
    #print(train_dataset[0])

if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, choices=("original", "ekman", "group"), help="Taxonomy (original, ekman, group)", default="original")
    cli_parser.add_argument("--model_type", type=str, choices=(GO_EMOTIONS, GO_EMOTIONS_MODIFIED, GO_EMOTIONS_SYNTAX_FED, GO_EMOTIONS_SYNTAX_INTERLEAVED), help=f'What model to use to train (\"{GO_EMOTIONS}\", \"{GO_EMOTIONS_MODIFIED}\", \"{GO_EMOTIONS_SYNTAX_FED}\", \"{GO_EMOTIONS_SYNTAX_INTERLEAVED}\")', default="goemotions")

    cli_args = cli_parser.parse_args()
    print("hello")
    main(cli_args)
