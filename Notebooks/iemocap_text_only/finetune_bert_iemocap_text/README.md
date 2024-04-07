---
tags:
- generated_from_trainer
model-index:
- name: finetune_bert_iemocap_text
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# finetune_bert_iemocap_text

This model was trained from scratch on an unknown dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.4672
- eval_accuracy: 0.8682
- eval_f1: 0.8691
- eval_runtime: 9.9975
- eval_samples_per_second: 55.414
- eval_steps_per_second: 7.002
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10

### Framework versions

- Transformers 4.38.2
- Pytorch 2.2.1
- Datasets 2.18.0
- Tokenizers 0.15.2
