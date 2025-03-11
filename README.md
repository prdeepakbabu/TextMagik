# BERT Subword Tokenization and MLM Example

This repository demonstrates a workflow that covers:

1. **Subword Tokenization**  
   - Examples of Byte-Level BPE, Character BPE, and WordPiece tokenization.

2. **Custom Tokenizer Training**  
   - Trains a subword tokenizer (default: WordPiece) on `.txt` files in the `data/` folder.

3. **BERT Model Training (Masked Language Modeling)**  
   - Builds and trains a small BERT model from scratch, using either the custom tokenizer or `bert-base-uncased` if no custom tokenizer is found.

4. **Embedding Visualization**  
   - Projects learned embeddings into a 3D space using t-SNE and displays them with matplotlib.

---

## Requirements

A `requirements.txt` might look like this:

```
torch>=1.10.0
transformers>=4.20.0
tokenizers>=0.10.0
scikit-learn>=0.24.0
numpy>=1.20.0
matplotlib>=3.5.0
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── data/
│   ├── sample1.txt
│   └── sample2.txt
├── main.py
├── requirements.txt
└── README.md
```

- **`data/`**: Contains the text files used for training the tokenizer and the BERT model.
- **`main.py`**: The main script that demonstrates tokenization, training, and visualization.
- **`requirements.txt`**: Lists the required Python packages.
- **`README.md`**: This file.

---

## Usage

1. **Prepare Your Data**  
   Place your training text files inside the `data/` folder. Each file should be in `.txt` format, containing one sample per line.

2. **Install Dependencies**  
   Ensure you have Python installed and run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Main Script**  
   Execute the following command to start the demonstration:
   ```bash
   python main.py
   ```
   This will perform the following tasks:
   - Demonstrate different subword tokenization methods.
   - Train a custom tokenizer on your text data.
   - Train a small BERT model using masked language modeling.
   - Visualize the token embeddings in a 3D plot using t-SNE.

4. **View the Outputs**  
   - The custom tokenizer files will be saved in the `tokenizer_model/` directory.
   - The trained BERT model and its tokenizer will be saved in the `bert_mlm_model/` directory.
   - A matplotlib window will pop up displaying the 3D visualization of token embeddings.

---

## Notes

- The script automatically detects GPU availability (CUDA or MPS) and uses it for training. If no GPU is detected, it falls back to CPU.
- The BERT model configuration is set for demonstration purposes with reduced parameters (e.g., fewer layers, smaller hidden size) to speed up training on small datasets.
- Modify the training parameters in `main.py` as needed to suit your dataset or desired model complexity.

---

## License

This project is provided under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for the model and training framework.
- [tokenizers](https://github.com/huggingface/tokenizers) for providing fast and flexible tokenization tools.
- [scikit-learn](https://scikit-learn.org/) for the t-SNE implementation used in visualization.