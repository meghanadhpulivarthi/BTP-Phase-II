from datasets import load_dataset, concatenate_datasets, DatasetDict, load_metric
from transformers import AutoTokenizer, DataCollatorWithPadding, pipeline
from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer, EarlyStoppingCallback
import argparse
import numpy as np
import wandb
import os
import torch
import pandas as pd

def add_args(parser):
    parser.add_argument("--do_train", action="store_true", help="Whether to train")
    parser.add_argument("--do_test", action="store_true", help="Whether to test")
    parser.add_argument("--do_loadmodel", action="store_true", help="Whether to load the saved model")
    parser.add_argument("--do_pushmodel", action="store_true", help="Whether to push the trained model")
    parser.add_argument("--out_dir", type=str, default="finetune_bert_iemocap_text", help="Where to save the model")
    parser.add_argument("--out_file", type=str, default="finetune_bert_iemocap_text.txt", help="Where to save the results")
    parser.add_argument("--run_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--train_batch_size", type=int, default=8, help="training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=8, help="testing and validation batch size")
    parser.add_argument("--epoch", type=int, default=20, help="How many epochs to run")
    parser.add_argument("--start_epoch", type=int, default=0, help="Which epcoh to start")
    args = parser.parse_args()
    return args

# def tokenize_function(examples, tokenizer):
#    return tokenizer(examples["text"], truncation=True)

def compute_metrics(eval_preds):
    load_accuracy = load_metric("accuracy")
    load_f1 = load_metric("f1")
  
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    accuracy = load_accuracy.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = load_f1.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    return {"accuracy": accuracy, "f1": f1}


def BERT_Trainer(args, id2label, label2id, checkpoint, tokenized_datasets, tokenizer, data_collator):
    training_args = TrainingArguments(
        args.out_dir, 
        evaluation_strategy="epoch", 
        num_train_epochs=args.epoch,
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        overwrite_output_dir=True,
        use_mps_device=True,
        push_to_hub=args.do_pushmodel,
        run_name=args.run_name,
    )

    num_labels = len(id2label)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, 
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=3 )],
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(pytorch_total_params)
    # wandb.log({"no_of_parameter": pytorch_total_params})

    trainer.train()
    trainer.evaluate()
    return trainer

def BERT(args):
    dataset = load_dataset("Zahra99/IEMOCAP_Text")
    dataset = concatenate_datasets([dataset["session1"], dataset["session2"], dataset["session3"], dataset["session4"], dataset["session5"]])

    # 90% train, 10% test + validation
    train_test_dataset = dataset.train_test_split(test_size=0.2)
    # Split the 10% test + valid in half test, half valid
    test_valid = train_test_dataset['test'].train_test_split(test_size=0.5)
    # gather everyone if you want to have a single DatasetDict
    train_test_valid_dataset = DatasetDict({
        'train': train_test_dataset['train']),
        'validation': test_valid['train'],
        'test': test_valid['test']})
    
    id2label_fn = train_test_valid_dataset["train"].features["label"].int2str

    checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = train_test_valid_dataset.map(lambda examples: tokenizer(examples["text"], truncation=True))
    # tokenized_datasets = train_test_valid_dataset.map(tokenize_function, batched=True, fn_kwargs=tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    if args.do_pushmodel == True:
        print(f"HuggingFace User is {os.system('huggingface-cli whoami')}")

    id2label = {
        str(i): id2label_fn(i)
        for i in range(len(tokenized_datasets["train"].features["label"].names))
    }
    label2id = {v: k for k, v in id2label.items()}
    assert torch.backends.mps.is_available() == True

    if args.do_train:
        trainer = BERT_Trainer(args, id2label, label2id, checkpoint, tokenized_datasets, tokenizer, data_collator)
        if args.do_test:
            print(f"IEMOCAP Test : {trainer.evaluate(eval_dataset=tokenized_datasets['test'])}")
            test_predictions = trainer.predict(tokenized_datasets["test"])
            test_preds = id2label_fn(np.argmax(test_predictions.predictions, axis=-1))
            test_gold = id2label_fn(test_predictions.label_ids)
            df = pd.DataFrame(list(zip(tokenized_datasets["test"]["text"], test_preds, test_gold)), 
                              columns=['Text', 'Predicted_label', 'Gold_label'])
            df.to_csv(args.out_file)


        if args.do_pushmodel:
            trainer.push_to_hub()
    
    if args.do_loadmodel:
        hub_model = pipeline("text-classification", model=f"meghanadh/{args.out_dir}")
        test_predictions = hub_model(tokenized_datasets["test"]["text"])
        test_preds = pd.DataFrame.from_records(test_predictions)['label'].tolist()
        test_gold = id2label_fn(tokenized_datasets["test"]["label"])
        df = pd.DataFrame(list(zip(tokenized_datasets["test"]["text"], test_preds, test_gold)), 
                              columns=['Text', 'Predicted_label', 'Gold_label'])
        df.to_csv(args.out_file)

def main():
    parser = argparse.ArgumentParser()
    args = add_args(parser)
    BERT(args)

if __name__ == "__main__":
    main()

