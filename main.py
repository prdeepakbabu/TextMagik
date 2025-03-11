#!/usr/bin/env python3
'''
main.py
main python script to
* different ways to tokenize text (subwords). show text and tokens generated. how to interpret them.
* train a text tokenizer based on a set of text files in data/ folder
* train a BERT model to learn representation based on MLM loss on GPU if available. (ex: mps or cuda) in pytorch/huggingface
* visualize embeddings of words in 3-d graph as vectors
'''
"""
main.py

Demonstration of:
1) Different methods to tokenize text (subwords). Show text and tokens.
2) Training a text tokenizer based on files in the data/ folder.
3) Training a small BERT model using MLM loss on GPU (if available) with PyTorch/Hugging Face.
4) Visualizing word embeddings in a 3D projection.
"""

import os
import glob
import torch
from tokenizers import ByteLevelBPETokenizer, CharBPETokenizer, BertWordPieceTokenizer
from transformers import (AutoConfig, BertForMaskedLM, BertTokenizerFast, DataCollatorForLanguageModeling,
                          Trainer, TrainingArguments)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


class TextDataset(Dataset):
    """
    A simple custom Dataset that reads all .txt files from a specified folder
    and provides text samples as individual lines.
    """
    def __init__(self, folder_path, tokenizer=None, max_length=128):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        # Load and store lines from all .txt files in the folder.
        txt_files = glob.glob(os.path.join(folder_path, '*.txt'))
        for txt_file in txt_files:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
                self.samples.extend(lines)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        # If a tokenizer is provided, tokenize here; otherwise just return text.
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
                return_tensors='pt'
            )
            # Convert from a batch of size 1 to a simple dict of tensors.
            return {k: v.squeeze(0) for k, v in encoding.items()}
        else:
            return text


def demonstrate_subword_tokenization(text="Hello world, how are you?"):
    """
    Demonstrates different subword tokenization strategies using the Hugging Face tokenizers library.
    """
    print("\n=== Demonstrating Subword Tokenization ===")
    print(f"Original text: '{text}'\n")

    # Byte-level BPE tokenizer
    bytelevel_bpe = ByteLevelBPETokenizer()
    # Train the tokenizer on just one example text (toy demonstration).
    bytelevel_bpe.train_from_iterator([text], vocab_size=50)
    bytelevel_output = bytelevel_bpe.encode(text)
    print("Byte-Level BPE:")
    print("Tokens:", bytelevel_output.tokens)
    print("IDs:   ", bytelevel_output.ids)
    print()

    # Character BPE tokenizer
    char_bpe = CharBPETokenizer()
    char_bpe.train_from_iterator([text], vocab_size=50)
    char_output = char_bpe.encode(text)
    print("Character BPE:")
    print("Tokens:", char_output.tokens)
    print("IDs:   ", char_output.ids)
    print()

    # BertWordPiece tokenizer
    wordpiece_tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False)
    wordpiece_tokenizer.train_from_iterator([text], vocab_size=50)
    wordpiece_output = wordpiece_tokenizer.encode(text)
    print("WordPiece:")
    print("Tokens:", wordpiece_output.tokens)
    print("IDs:   ", wordpiece_output.ids)
    print()


def train_custom_tokenizer(data_folder="data", vocab_size=3000, tokenizer_type="wordpiece"):
    """
    Train a custom tokenizer on all .txt files in the data_folder.
    """
    print("\n=== Training a custom tokenizer ===")
    txt_files = glob.glob(os.path.join(data_folder, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in '{data_folder}'. Please add text files to train the tokenizer.")
        return None

    # Read all lines from the data folder.
    all_lines = []
    for path in txt_files:
        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            all_lines.extend(lines)

    if tokenizer_type == "bytebpe":
        tokenizer = ByteLevelBPETokenizer()
    elif tokenizer_type == "charbpe":
        tokenizer = CharBPETokenizer()
    else:
        tokenizer = BertWordPieceTokenizer(clean_text=True, handle_chinese_chars=False)

    tokenizer.train_from_iterator(all_lines, vocab_size=vocab_size)

    # Save or return the tokenizer for later use.
    tokenizer.save_model("tokenizer_model")
    print(f"Tokenizer of type '{tokenizer_type}' trained and saved to 'tokenizer_model'.")

    return tokenizer


def train_bert_model(data_folder="data", pretrained_tokenizer_path=None, epochs=1, batch_size=8):
    """
    Train a BERT model from scratch (small config) on the data in data_folder using MLM.
    """
    print("\n=== Training a BERT model on MLM ===")

    # If a pretrained custom tokenizer is available, use it.
    # Otherwise, use 'bert-base-uncased' as a fallback.
    if pretrained_tokenizer_path:
        # Hugging Face's BertTokenizerFast can load from a directory containing vocab files.
        tokenizer = BertTokenizerFast.from_pretrained(pretrained_tokenizer_path, do_lower_case=True)
        print(f"Loaded tokenizer from {pretrained_tokenizer_path}.")
    else:
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        print("Using 'bert-base-uncased' tokenizer.")

    # Prepare dataset
    train_dataset = TextDataset(data_folder, tokenizer=tokenizer)
    if len(train_dataset) == 0:
        print("No data found for training. Aborting.")
        return

    # Build a small BERT config or use the default config from 'bert-base-uncased'
    config = AutoConfig.from_pretrained("bert-base-uncased")
    config.hidden_size = 128  # smaller hidden size for demonstration
    config.num_hidden_layers = 4  # fewer layers
    config.num_attention_heads = 4  # fewer attention heads
    config.intermediate_size = 256  # smaller intermediate size

    model = BertForMaskedLM(config)

    # Check for GPU (CUDA or MPS)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else
                          "cpu")
    model.to(device)
    print(f"Using device: {device}")

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="bert_mlm_output",
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=500,
        save_total_limit=2,
        logging_steps=50,
        evaluation_strategy="no"  # For demonstration, turn off evaluation
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    # Train the model
    trainer.train()
    print("Training complete.")

    # Save the model
    trainer.save_model("bert_mlm_model")
    tokenizer.save_pretrained("bert_mlm_model")
    print("Model and tokenizer saved to 'bert_mlm_model'.")


def visualize_embeddings(tokenizer_path="bert_mlm_model", num_tokens=50):
    """
    Visualize word/token embeddings in a 3D projection using TSNE.
    """
    print("\n=== Visualizing Embeddings in 3D ===")

    # Load model and tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(tokenizer_path)
    model.eval()

    # We'll gather embeddings from the model's input embedding layer.
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # Let's pick the first `num_tokens` tokens from the tokenizer vocabulary (excluding special ones if possible)
    # so we don't clutter the visualization.
    # We'll gather their embeddings and project them.

    # We can filter out special tokens by checking if token isn't [PAD], [UNK], etc.
    # But for simplicity, let's just pick the first N tokens.
    vocab_size = embedding_matrix.shape[0]
    max_tokens = min(num_tokens, vocab_size)

    tokens_to_visualize = list(range(9999))  # Just a large range
    tokens_to_visualize = tokens_to_visualize[:max_tokens]

    sub_matrix = embedding_matrix[tokens_to_visualize]
    # TSNE to 3D
    tsne = TSNE(n_components=3, random_state=42)
    reduced_embeddings = tsne.fit_transform(sub_matrix)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2]
    )

    # Annotate points with their token strings if desired
    for i, token_id in enumerate(tokens_to_visualize):
        token_str = tokenizer.convert_ids_to_tokens(token_id)
        ax.text(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            reduced_embeddings[i, 2],
            token_str
        )

    plt.title("3D Visualization of Token Embeddings (TSNE)")
    plt.show()


if __name__ == "__main__":
    # 1) Demonstrate subword tokenization
    demonstrate_subword_tokenization("Hello world, how are you today? This is a test.")

    # 2) Train a custom tokenizer from text files in data/ folder.
    #    For demonstration, use WordPiece.
    custom_tokenizer = train_custom_tokenizer(
        data_folder="data",
        vocab_size=3000,
        tokenizer_type="wordpiece"
    )

    # 3) Train a small BERT model with masked language modeling.
    #    Use the newly created tokenizer if it was created successfully.
    if custom_tokenizer:
        # Hugging Face expects the saved tokenizer to be in a folder with vocab.txt, etc.
        # We saved it to "tokenizer_model/", so pass that path.
        pretrained_path = "tokenizer_model"
    else:
        pretrained_path = None

    train_bert_model(
        data_folder="data",
        pretrained_tokenizer_path=pretrained_path,
        epochs=1,
        batch_size=8
    )

    # 4) Visualize embeddings of words in 3D.
    #    Use the newly trained BERT model and tokenizer in "bert_mlm_model".
    visualize_embeddings(
        tokenizer_path="bert_mlm_model",
        num_tokens=50
    )

    print("\nAll tasks completed.")