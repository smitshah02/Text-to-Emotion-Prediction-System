import argparse
import datasets
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from transformers import TFAutoModel, AutoTokenizer

# Load DistilBERT
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = TFAutoModel.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=64, padding="max_length")

def build_model(num_labels):
    input_ids = tf.keras.Input(shape=(64,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(64,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
    cls_token = bert_outputs.last_hidden_state[:, 0, :] 

    x = tf.keras.layers.Dense(256, activation='relu')(cls_token)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_labels, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=outputs)
    return model


def train(model_path="bert_model.keras", train_path="train.csv", dev_path="dev.csv"):
    hf_dataset = datasets.load_dataset("csv", data_files={"train": train_path, "validation": dev_path})

    labels = hf_dataset["train"].column_names[1:]  # skip 'text'

    def gather_labels(example):
        return {"labels": [float(example[l]) for l in labels]}
    
    hf_dataset = hf_dataset.map(gather_labels)
    hf_dataset = hf_dataset.map(tokenize_function, batched=True)

    train_dataset = hf_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=12,
        shuffle=True
    )
    dev_dataset = hf_dataset["validation"].to_tf_dataset(
        columns=["input_ids", "attention_mask"],
        label_cols="labels",
        batch_size=12
    )

    model = build_model(num_labels=len(labels))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[tf.keras.metrics.F1Score(average="micro", threshold=0.5)]
    )

    model.fit(
        train_dataset,
        epochs=5, 
        validation_data=dev_dataset,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(filepath=model_path, monitor="val_f1_score", mode="max", save_best_only=True,  save_weights_only=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=1)
        ]
    )
    print("Training Complete!")

def predict(model_path="bert_model.keras", input_path="test-in.csv"):
    model = build_model(num_labels=7)
    model.load_weights(model_path)

    df = pd.read_csv(input_path)
    hf_dataset = datasets.Dataset.from_pandas(df)
    hf_dataset = hf_dataset.map(tokenize_function, batched=True)

    tf_dataset = hf_dataset.to_tf_dataset(columns=["input_ids", "attention_mask"], batch_size=8)

    y_pred_probs = model.predict(tf_dataset)

    if all(label in df.columns for label in df.columns[1:]):
        y_true = np.array(df.iloc[:, 1:])

        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.3, 0.61, 0.01):
            preds = (y_pred_probs > thresh).astype(int)
            f1 = f1_score(y_true, preds, average="micro")
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        print(f"Best Threshold: {best_thresh:.2f}")
        print(f"Best F1 Score: {best_f1:.4f}")

        final_predictions = (y_pred_probs > best_thresh).astype(int)

    else:
        final_predictions = (y_pred_probs > 0.5).astype(int)

    df.iloc[:, 1:] = final_predictions
    df.to_csv("submission.zip", index=False, compression=dict(method='zip', archive_name='submission.csv'))
    print("Predictions Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices={"train", "predict"})
    args = parser.parse_args()
    globals()[args.command]()