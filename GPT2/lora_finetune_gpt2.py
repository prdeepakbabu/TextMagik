import os
import math
import random

import torch
from torch.utils.data import random_split
from datasets import load_dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2TokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
import wandb
import matplotlib.pyplot as plt  # <-- Added for plotting
import wandb
from dotenv import load_dotenv

load_dotenv()  # This will read in variables from .env
# Optionally, read the key directly:
api_key = os.getenv('WANDB_API_KEY')


def run_inference(model, tokenizer, prompt, max_new_tokens=50):
    """
    Generates text from the fine-tuned model given a prompt.
    """
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


class TestLossPlotCallback(TrainerCallback):
    """
    A callback to track and plot eval (test) loss over steps.
    """
    def __init__(self):
        super().__init__()
        self.eval_steps = []
        self.eval_losses = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Each time evaluation is run, store the global_step and eval_loss.
        """
        if metrics and "eval_loss" in metrics:
            step = state.global_step
            loss_val = metrics["eval_loss"]
            self.eval_steps.append(step)
            self.eval_losses.append(loss_val)

    def on_train_end(self, args, state, control, **kwargs):
        """
        After training completes, plot the recorded eval_loss values.
        """
        if len(self.eval_steps) > 0:
            plt.figure()
            plt.plot(self.eval_steps, self.eval_losses, marker='o', label='Eval Loss')
            plt.xlabel('Global Step')
            plt.ylabel('Loss')
            plt.title('Evaluation Loss Over Steps')
            plt.legend()
            # Save the plot to your output_dir
            plot_path = os.path.join(args.output_dir, "eval_loss_plot.png")
            plt.savefig(plot_path)
            plt.show()
            print(f"Eval loss plot saved to {plot_path}")


def main(
    train_file="medical_chat_processed.txt",
    output_dir="lora_gpt2_medical",
    num_train_epochs=3,
    train_batch_size=2,
    eval_batch_size=2,
    lr=1e-4,
    logging_steps=50
):
    # -------------------
    # 1) Check device (MPS, CUDA, or CPU)
    # -------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device.")
    else:
        device = torch.device("cpu")
        print("Using CPU device.")

    # -------------------
    # 2) Initialize W&B
    # -------------------
    wandb.init(
        project="gpt2-lora-medical",
        config={
            "train_file": train_file,
            "output_dir": output_dir,
            "num_train_epochs": num_train_epochs,
            "train_batch_size": train_batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": lr,
        }
    )

    # -------------------
    # 3) Load dataset
    # -------------------
    dataset = load_dataset("text", data_files={"train": train_file})["train"]
    # Shuffle for randomness
    dataset = dataset.shuffle(seed=42)

    # 10% for test set
    test_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size
    #train_dataset, eval_dataset = random_split(dataset, [train_size, test_size])
    # 1) Load and shuffle
    dataset = load_dataset("text", data_files={"train": train_file})["train"]
    dataset = dataset.shuffle(seed=42)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Use eos_token for padding to avoid size mismatch
    tokenizer.pad_token = tokenizer.eos_token


    # 2) Use train_test_split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # 3) Now tokenize
    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=256)

    train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # -------------------
    # 4) Load tokenizer & model
    # -------------------
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # Use eos_token for padding to avoid size mismatch
    tokenizer.pad_token = tokenizer.eos_token

    base_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # -------------------
    # 5) Prepare LoRA
    # -------------------
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["c_attn", "c_fc"],  # GPT-2 attention/FC modules
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base_model, lora_config)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # -------------------
    # 6) Tokenize dataset
    # -------------------
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=256
        )

    #train_dataset = train_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    #eval_dataset = eval_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

    # -------------------
    # 7) Data Collator
    # -------------------
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # -------------------
    # 8) Training Arguments
    # -------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=lr,
        logging_steps=logging_steps,
        evaluation_strategy="steps",  # Evaluate every `logging_steps` steps
        save_strategy="epoch",
        logging_dir=os.path.join(output_dir, "logs"),
        report_to=["wandb"],  # log metrics to wandb
        fp16=False,           # MPS doesn't support FP16
        bf16=False,
    )

    # -------------------
    # 9) Trainer + Callback
    # -------------------
    test_loss_callback = TestLossPlotCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[test_loss_callback]
    )

    # -------------------
    # 10) Train
    # -------------------
    trainer.train()

    # -------------------
    # 11) Evaluate
    # -------------------
    eval_metrics = trainer.evaluate()
    print("Evaluation results:", eval_metrics)
    wandb.log(eval_metrics)

    # Save final model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # -------------------
    # 12) Example inference
    # -------------------
    sample_prompt = "[PATIENT]: I've had a severe headache for 2 days.\n[DOCTOR]:"
    generated_text = run_inference(model, tokenizer, sample_prompt, max_new_tokens=50)
    print("\n=== Sample Inference ===")
    print(generated_text)

    wandb.finish()


if __name__ == "__main__":
    main(
        train_file="../data/medical_chat_processed.txt",
        output_dir="lora_gpt2_medical",
        num_train_epochs=1,
        train_batch_size=32,
        eval_batch_size=32,
        lr=1e-4,
        logging_steps=500
    )