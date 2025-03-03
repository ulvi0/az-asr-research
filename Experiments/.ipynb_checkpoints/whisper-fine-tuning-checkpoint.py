import os
import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import load_dataset, Audio
import pandas as pd
def main():
    # Specify the model name (using the large version)
    model_name = "openai/whisper-large-v3"

    # Load the processor and model from Hugging Face
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    # Create a data collator to dynamically pad the input features and labels in each batch.
    model.gradient_checkpointing_enable()
    def data_collator(features):
        # Extract input features and labels lists from the batch
        input_features = [f["input_features"] for f in features]
        input_features = {"input_features": input_features}
        labels = [f["labels"] for f in features]
        # Pad input features (using the feature extractor's padding method)`
        batch_inputs = processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad labels using the tokenizer's pad method.
        batch_labels = processor.tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]

        return {"input_features": batch_inputs.input_features, "labels": batch_labels}
    def prepare_example(batch):
        # "audio_path" is automatically loaded as a dictionary with an "array" key.
        audio = batch["audio_path"]["array"]
        transcript = batch["transcript"]

        # Extract features from the audio using the processor's feature extractor.
        # The result is a list with one element per audio sample.
        input_features = processor.feature_extractor(audio, sampling_rate=16000).input_features[0]

        # Tokenize the transcript (the tokenizer will handle any necessary preprocessing).
        labels = processor.tokenizer(transcript).input_ids

        # Store the processed features and labels in the batch.
        batch["input_features"] = input_features
        batch["labels"] = labels
        return batch
    # Load your custom dataset from CSV files.
    # Ensure your CSV files have at least two columns: "audio_path" and "transcript".
    data_files = {"train": "train.csv", "validation": "val.csv"}
    dataset = load_dataset("csv", data_files=data_files)

    # Cast the "audio_path" column to an Audio column with the desired sampling rate.
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))

    # Apply the preprocessing function to all examples.
    # remove_columns will drop the original columns so that the model receives only what it needs.
    dataset = dataset.map(prepare_example, remove_columns=dataset["train"].column_names)
    # Define the training arguments.
    # Adjust the batch sizes, learning rate, number of epochs, etc. according to your needs.
    training_args = Seq2SeqTrainingArguments(
        output_dir="./whisper-finetuned",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        evaluation_strategy="steps",
        num_train_epochs=3,
        bf16=True,  # enable this if you have a GPU that supports half precision
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=1e-5,
        predict_with_generate=True,
        logging_dir="./logs",
        fp16=False
        )

    # Create a Trainer for sequence-to-sequence training.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=data_collator,
        tokenizer=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=True)


    # After training is complete, save the fine-tuned model and processor
    #model.save_pretrained("./whisper-finetuned")
    processor.save_pretrained("./whisper-finetuned")


if __name__ == "__main__":
    main()

