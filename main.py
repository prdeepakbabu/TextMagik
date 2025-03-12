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
                          Trainer, TrainingArguments, TrainerCallback)
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import wandb
from dotenv import load_dotenv

load_dotenv()  # This will read in variables from .env
# Optionally, read the key directly:
api_key = os.getenv('WANDB_API_KEY')


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


def train_bert_model(data_folder="data", wandb_entity = None,pretrained_tokenizer_path=None, epochs=1, batch_size=8):
    """
    Train a BERT model from scratch (small config) on the data in data_folder using MLM.
    """
    print("\n=== Training a BERT model on MLM ===")

    class LossPlotCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.train_loss_history = []
            self.steps_history = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            # logs typically contains 'loss', 'learning_rate', etc.
            if 'loss' in logs:
                self.train_loss_history.append(logs['loss'])
                self.steps_history.append(state.global_step)

    # Initialize Weights & Biases for logging (W&B offers a free tier for individual use)
    if wandb_entity:
        wandb.init(project="bert-mlm-training", entity=wandb_entity, config={"epochs": epochs, "batch_size": batch_size})
    else:
        wandb.init(project="bert-mlm-training", config={"epochs": epochs, "batch_size": batch_size})

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
        evaluation_strategy="no",  # For demonstration, turn off evaluation
        report_to=["wandb"]
    )

    loss_callback = LossPlotCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        callbacks=[loss_callback]
    )

    # Train the model
    trainer.train()

    # Plot the loss after training
    plt.figure()
    plt.plot(loss_callback.steps_history, loss_callback.train_loss_history, label='Training Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Steps')
    plt.legend()
    plt.show()

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



def visualize_specific_words(tokenizer_path="bert_mlm_model", words=None):
    """
    Visualize specific user-provided words in a 3D projection using TSNE.
    Even if a word splits into multiple sub-tokens, we average those embeddings
    to represent the entire word.
    """
    if not words:
        print("No words provided. Please pass a list of words to visualize.")
        return

    print("\n=== Visualizing Specific Words in 3D ===")
    print("Words:", words)

    # 1. Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(tokenizer_path)
    model.eval()

    # 2. Get the embedding matrix
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    # 3. For each user word, tokenize into subwords, then average their embeddings
    valid_words = []
    word_vectors = []

    for w in words:
        # Convert the word into subword tokens
        sub_tokens = tokenizer.tokenize(w)
        if not sub_tokens:
            print(f"'{w}' yielded no sub tokens. Skipping.")
            continue

        # Convert sub-tokens to IDs
        sub_token_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        # Gather each sub-token's embedding
        sub_embeddings = embedding_matrix[sub_token_ids]  # shape: (num_subtokens, hidden_dim)
        # Average them to get a single vector for the entire word
        avg_embedding = sub_embeddings.mean(axis=0)

        word_vectors.append(avg_embedding)
        valid_words.append(w)

    if not word_vectors:
        print("No valid tokens found for the given words.")
        return

    word_vectors = np.array(word_vectors)
    n_samples = word_vectors.shape[0]

    # 4. TSNE to 3D
    # Perplexity must be < n_samples, so pick a smaller perplexity if needed
    perplexity = min(30, max(2, n_samples - 1))
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(word_vectors)

    # 5. Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        reduced_embeddings[:, 2]
    )

    # Annotate each point with the corresponding word
    for i, word in enumerate(valid_words):
        ax.text(
            reduced_embeddings[i, 0],
            reduced_embeddings[i, 1],
            reduced_embeddings[i, 2],
            word
        )

    plt.title("3D Visualization of Specific Words (TSNE)")
    plt.show()

def summarize_data_statistics(data_folder="data"):
    """
    Reads all .txt files in the given data_folder and prints summary statistics:
    total lines, total words, total sentences.
    """
    import glob, os
    total_lines = 0
    total_words = 0
    total_sentences = 0

    txt_files = glob.glob(os.path.join(data_folder, '*.txt'))
    for txt_file in txt_files:
        with open(txt_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                total_lines += 1
                words_in_line = line.split()
                total_words += len(words_in_line)
                sentences_in_line = [s for s in line.split('.') if s.strip()]
                total_sentences += len(sentences_in_line)

    print("\n=== Data Summary ===")
    print(f"Total lines: {total_lines}")
    print(f"Total words: {total_words}")
    print(f"Total sentences (naive): {total_sentences}")
    print("====================\n")


def visualize_word_pairs_3d(tokenizer_path="bert_mlm_model", pairs=None):
    """
    Visualize word pairs in 3D using TSNE. Each pair is drawn as a line
    from the first word's point to the second word's point.

    :param tokenizer_path: Path to the trained BERT tokenizer & model.
    :param pairs: List of (word1, word2) tuples, e.g. [("ship", "water"), ("car", "road")]
    """
    if not pairs:
        print("No pairs provided. Please pass a list of (word1, word2) tuples.")
        return

    # 1. Gather all unique words from the pairs
    unique_words = set()
    for (w1, w2) in pairs:
        unique_words.add(w1)
        unique_words.add(w2)
    unique_words = list(unique_words)  # fix an order

    print("\n=== Visualizing Word Pairs in 3D ===")
    print("Word Pairs:", pairs)

    # 2. Load tokenizer and model
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    model = BertForMaskedLM.from_pretrained(tokenizer_path)
    model.eval()

    # 3. Extract embeddings for each unique word
    #    If a word splits into multiple subwords, we'll average them to get a single vector.
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    def get_average_embedding(word):
        sub_tokens = tokenizer.tokenize(word)
        if not sub_tokens:
            return None
        sub_ids = tokenizer.convert_tokens_to_ids(sub_tokens)
        sub_embs = embedding_matrix[sub_ids]  # shape (num_subtokens, hidden_dim)
        return sub_embs.mean(axis=0)

    word_vectors = []
    valid_words = []

    for w in unique_words:
        vec = get_average_embedding(w)
        if vec is not None:
            word_vectors.append(vec)
            valid_words.append(w)
        else:
            print(f"'{w}' not found in vocabulary (will be skipped).")

    if len(word_vectors) < 2:
        print("Not enough valid words to plot.")
        return

    word_vectors = np.array(word_vectors)
    n_samples = word_vectors.shape[0]

    # 4. TSNE in 3D
    #    If perplexity is too large for a small set, reduce it:
    from math import floor
    perplexity = min(30, max(2, n_samples - 1))
    tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
    coords_3d = tsne.fit_transform(word_vectors)  # shape [n_samples, 3]

    # 5. Plot each point
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2], color='blue')

    # Build a dict to map each valid word -> its 3D coordinates
    word_to_coords = {w: coords_3d[i] for i, w in enumerate(valid_words)}

    # Label each point
    for i, w in enumerate(valid_words):
        x, y, z = coords_3d[i]
        ax.text(x, y, z, w, fontsize=9)

    # 6. Draw lines for each pair
    for (w1, w2) in pairs:
        if w1 in word_to_coords and w2 in word_to_coords:
            x1, y1, z1 = word_to_coords[w1]
            x2, y2, z2 = word_to_coords[w2]

            # Plot a dashed line from w1 to w2
            ax.plot(
                [x1, x2],
                [y1, y2],
                [z1, z2],
                linestyle='dashed',
                color='gray'
            )

    ax.set_title("3D Word Pair Visualization (TSNE)")
    plt.show()

if __name__ == "__main__":
    # 1) Demonstrate subword tokenization
    #demonstrate_subword_tokenization("Hello world, how are you today? This is a test.")

    # 2) Summarize data
    summarize_data_statistics("data")
    '''
    # 3) Train a custom tokenizer from text files in data/ folder.
    #    For demonstration, use WordPiece.
    custom_tokenizer = train_custom_tokenizer(
        data_folder="data",
        vocab_size=3000,
        tokenizer_type="wordpiece"
    )

    # 4) Train a small BERT model with masked language modeling.
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
        epochs=10,
        batch_size=8
    )

    # 5) Visualize embeddings of words in 3D.
    #    Use the newly trained BERT model and tokenizer in "bert_mlm_model".
    visualize_embeddings(
        tokenizer_path="bert_mlm_model",
        num_tokens=50
    )'''

    # 6) Visualize a few user-provided words in 3D
    words_to_check = ["suspense","horror","thriller"]
    #visualize_specific_words(
    #    tokenizer_path="bert_mlm_model",
    #    words=words_to_check
    #)

    # Example usage:
    pairs_to_plot = [("him", "he"), ("king", "queen"),("man","women")]
    visualize_word_pairs_3d(
        tokenizer_path="bert_mlm_model",
        pairs=pairs_to_plot
    )
    print("\nAll tasks completed.")