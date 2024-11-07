from datasets import Dataset
import evaluate
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification

# Load datasets
train_ds = Dataset.load_from_disk('data/train_dataset')
test_ds = Dataset.load_from_disk('data/test_dataset')

label_list = ["O", "B-mountain", "I-mountain"]
label2id = dict(map(lambda i: (label_list[i], i), range(3)))
id2label = dict(map(lambda i: (i, label_list[i]), range(3)))

# Metrics for evaluation
seqeval = evaluate.load("seqeval")
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# Load model
model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-multilingual-cased", id2label=id2label, label2id=label2id, finetuning_task="ner"
)

training_args = TrainingArguments(
    output_dir="mountain_ner_model",
    num_train_epochs=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    # push_to_hub=True,
)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-multilingual-cased")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)  # dynamically pad the sentences

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()
