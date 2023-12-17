import argparse
import json
import logging
import os
import glob
import csv
import time

import numpy as np
#from stanford
#from syntax_interleaved_bert_multi_label_classifier import SyntaxInterleavedBertForMultiLabelClassification-parser.load_trees as load_trees_from
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import (
    BertConfig,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)

from bert_multi_label_classifier import BertForMultiLabelClassification
from modified_multi_label_classifier import ModifiedBertForMultiLabelClassification
from syntax_bert_multi_label_classifier import SyntaxFedBertForMultiLabelClassification
from syntax_interleaved_bert_multi_label_classifier import SyntaxInterleavedBertForMultiLabelClassification
from utils import (
    init_logger,
    set_seed,
    compute_metrics
)
from data_loader import (
    load_and_cache_examples,
    GoEmotionsProcessor
)
from data_loader_with_trees import (
    load_and_cache_examples_with_trees,
    GoEmotionsProcessorWithTrees
)

CSV_FILE = os.path.join(".", "evals.csv")

logger = logging.getLogger(__name__)


def train(args,
          model,
          tokenizer,
          train_dataset,
          model_type,
          run_id,
          dev_dataset,
          test_dataset):
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(t_total * args.warmup_proportion),
        num_training_steps=t_total
    )

    if os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and os.path.isfile(
            os.path.join(args.model_name_or_path, "scheduler.pt")
    ):
        # Load optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Logging steps = %d", args.logging_steps)
    logger.info("  Save steps = %d", args.save_steps)

    global_step = 0
    tr_loss = 0.0

    start_time = time.time() 

    # the global steps at which the model was saved
    saved_steps = []

    model.zero_grad()
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    for epoch in train_iterator:
        epoch += 1
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            outputs = model(
                input_ids      = batch[0],
                attention_mask = batch[1],
                token_type_ids = batch[2],
                labels         = batch[3]
            )

            loss = outputs[0]

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    len(train_dataloader) <= args.gradient_accumulation_steps
                    and (step + 1) == len(train_dataloader)
            ):
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                model.zero_grad()
                global_step += 1

            if args.max_steps > 0 and global_step > args.max_steps:
                break

        # Save model checkpoint
        output_dir = os.path.join(args.output_dir, "checkpoint-{}-{}".format(model_type, global_step))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        model_to_save = (model.module if hasattr(model, "module") else model)
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        saved_steps.append(str(global_step))

        torch.save(args, os.path.join(output_dir, "training_args.bin"))
        logger.info("Saving model checkpoint to {}".format(output_dir))

        if args.save_optimizer:
            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to {}".format(output_dir))

        # Evaluate the model at this epoch
        time_passed = time.time() - start_time
        #evaluate(args, model, test_dataset, "test", run_id, time_passed, True, model_type, epoch, global_step)
        evaluate(args, model, dev_dataset , "dev" , run_id, time_passed, True, model_type, epoch, args.adjacency_threshhold, args.tree_threshhold, global_step)

        if args.max_steps > 0 and global_step > args.max_steps:
            break

    return saved_steps, global_step, tr_loss / global_step


def evaluate(args, model, eval_dataset, mode, run_id, time_taken, add_csv_row, model_type, epoch, adjacency_threshhold, tree_threshhold, global_step=None):
    results = {}
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Eval!
    if global_step != None:
        logger.info("***** Running evaluation on {} dataset ({} step) *****".format(mode, global_step))
    else:
        logger.info("***** Running evaluation on {} dataset *****".format(mode))
    logger.info("  Num examples = {}".format(len(eval_dataset)))
    logger.info("  Eval Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        labels = batch[3]

        with torch.no_grad():
            outputs = model(
                input_ids      = batch[0],
                attention_mask = batch[1],
                token_type_ids = batch[2],
                labels         = batch[3]
            )
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()

        nb_eval_steps += 1
        if preds is None:
            preds = 1 / (1 + np.exp(-logits.detach().cpu().numpy()))  # Sigmoid
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, 1 / (1 + np.exp(-logits.detach().cpu().numpy())), axis=0)  # Sigmoid
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    results = {
        "loss": eval_loss
    }
    preds[preds > args.threshold] = 1
    preds[preds <= args.threshold] = 0
    (result, headers, values) = compute_metrics(out_label_ids, preds)
    results.update(result)

    headers = ["run_id",     "seed", "epoch", "mode",      "taxonomy", "time_taken", "model_type"] + headers + ["adjacency_threshhold", "tree_threshhold"]
    values  = [ run_id , args.seed ,  epoch ,  mode ,  args.taxonomy ,  time_taken ,  model_type ] + values  + [ adjacency_threshhold ,  tree_threshhold ]

    if add_csv_row:
        if not os.path.exists(CSV_FILE):
            # create the file and add header row to it
            with open(CSV_FILE, 'w+') as f: # create the file
                writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                writer.writerow(headers)
                writer.writerow(values)
        else:
            # just append the values to the csv
            with open(CSV_FILE, mode="a", encoding="utf-8") as f:
                writer = csv.writer(f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL, lineterminator="\n")
                writer.writerow(values)

    output_dir = os.path.join(args.output_dir, mode)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_eval_file = os.path.join(output_dir, "{}-{}.txt".format(mode, global_step) if global_step else "{}.txt".format(mode))
    with open(output_eval_file, "w") as f_w:
        logger.info("***** Eval results on {} dataset *****".format(mode))
        for key in sorted(results.keys()):
            logger.info("  {} = {}".format(key, str(results[key])))
            f_w.write("  {} = {}\n".format(key, str(results[key])))

    return results

GO_EMOTIONS = "goemotions"
GO_EMOTIONS_MODIFIED = "goemotions_modified"
GO_EMOTIONS_SYNTAX_FED = "goemotions_syntax_fed"
GO_EMOTIONS_SYNTAX_INTERLEAVED = "goemotions_syntax_interleaved"

def main(cli_args):
    # Read from config file and make args
    config_filename = "{}.json".format(cli_args.taxonomy)
    with open(os.path.join("data", "config", config_filename)) as f:
        args = argparse.Namespace(**json.load(f))
    logger.info("Training/evaluation parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)
    args.taxonomy = cli_args.taxonomy
    args.model_type = cli_args.model_type

    if not hasattr(args, "adjacency_threshhold"):
        args.adjacency_threshhold = 0
    if not hasattr(args, "tree_threshhold"):
        args.tree_threshhold = 0

    init_logger() 

    import random
    random.seed(None) # make sure that `random` uses system time for randomness
    args.seed = random.randint(0, (2**32)-1) # get a randomly-generated seed
    set_seed(args) # set the seed for all objects (including `random`)

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
    if args.model_type == GO_EMOTIONS:
        model = BertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
    elif args.model_type == GO_EMOTIONS_MODIFIED:
        model = ModifiedBertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
    elif args.model_type == GO_EMOTIONS_SYNTAX_FED:
        model = SyntaxFedBertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
    elif args.model_type == GO_EMOTIONS_SYNTAX_INTERLEAVED:
        model = SyntaxInterleavedBertForMultiLabelClassification.from_pretrained(
            args.model_name_or_path,
            config=config
        )
    else:
        raise NotImplementedError(f"{args.model_type} is an unknown model option")

    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    model.to(args.device)
    print("device: ", args.device)

    run_id = str(time.time())

    if args.model_type == GO_EMOTIONS_SYNTAX_INTERLEAVED: # change to just syntax interleaved model later
        train_dataset = load_and_cache_examples_with_trees(args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = load_and_cache_examples_with_trees(args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = load_and_cache_examples_with_trees(args, tokenizer, mode="test") if args.test_file else None
    else:
        train_dataset = load_and_cache_examples(args, tokenizer, mode="train") if args.train_file else None
        dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev") if args.dev_file else None
        test_dataset = load_and_cache_examples(args, tokenizer, mode="test") if args.test_file else None

    saved_steps, global_step, tr_loss = train(args, model, tokenizer, train_dataset, args.model_type, run_id, dev_dataset, test_dataset)
    logger.info(" global_step = {}, average loss = {}".format(global_step, tr_loss))

    results = {}
    if args.do_eval:
        checkpoints = [os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))]
        checkpoints = [checkpoint for checkpoint in checkpoints if (args.model_type in checkpoint)]
        if not args.eval_all_checkpoints:
            checkpoints = checkpoints[-1:]
        else:
            logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            split = checkpoint.split("-")
            (model_type, global_step) = (split[-2], split[-1])

            if str(global_step) not in saved_steps:
                # only evaluate the steps that were saved during training
                continue

            if args.model_type == GO_EMOTIONS:
                model = BertForMultiLabelClassification.from_pretrained(checkpoint)
            elif args.model_type == GO_EMOTIONS_MODIFIED:
                model = ModifiedBertForMultiLabelClassification.from_pretrained(checkpoint)
            elif args.model_type == GO_EMOTIONS_SYNTAX_FED:
                model = SyntaxFedBertForMultiLabelClassification.from_pretrained(checkpoint)
            elif args.model_type == GO_EMOTIONS_SYNTAX_INTERLEAVED:
                model = SyntaxInterleavedBertForMultiLabelClassification.from_pretrained(checkpoint)
            else:
                raise NotImplementedError(f"{args.model_type} is an unknown model option")

            model.to(args.device)
            result = evaluate(args, model, test_dataset, "test", run_id=run_id, time_taken=0, add_csv_row=True, model_type=args.model_type, epoch=0, adjacency_threshhold=args.adjacency_threshhold, tree_threshhold=args.tree_threshhold, global_step=global_step)
            result = dict((k + "_{}".format(global_step), v) for k, v in result.items())
            results.update(result)

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as f_w:
            for key in sorted(results.keys()):
                f_w.write("{} = {}\n".format(key, str(results[key])))


if __name__ == '__main__':
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--taxonomy", type=str, choices=("original", "ekman", "group"), help="Taxonomy (original, ekman, group)", default="original")
    cli_parser.add_argument("--model_type", type=str, choices=(GO_EMOTIONS, GO_EMOTIONS_MODIFIED, GO_EMOTIONS_SYNTAX_FED, GO_EMOTIONS_SYNTAX_INTERLEAVED), help=f'What model to use to train (\"{GO_EMOTIONS}\", \"{GO_EMOTIONS_MODIFIED}\", \"{GO_EMOTIONS_SYNTAX_FED}\", \"{GO_EMOTIONS_SYNTAX_INTERLEAVED}\")', default="goemotions")

    cli_args = cli_parser.parse_args()
    main(cli_args)
    main(cli_args)
    main(cli_args)
