import os
import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def load_model(model_name_or_path):
    # Load the GPT2 pretrained model and tokenizer from Hugging Face
    model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name_or_path)

    model.eval()
    # Determine device: check for CUDA, then for Apple's MPS (Apple Silicon), else CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)
    return model, tokenizer, device


def generate_text(model, tokenizer, prompt, decoding_strategy="sampling",
                  temperature=1.0, max_new_tokens=50, num_beams=5, device="cpu"):
    # Tokenize the prompt and move to the appropriate device
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

    # Prepare generation arguments based on the decoding strategy
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }

    if decoding_strategy == "greedy":
        gen_kwargs["do_sample"] = False
    elif decoding_strategy == "beam":
        gen_kwargs["num_beams"] = num_beams
        gen_kwargs["do_sample"] = False
    elif decoding_strategy == "sampling":
        gen_kwargs["do_sample"] = True
    else:
        print("Unknown decoding strategy, defaulting to sampling.")
        gen_kwargs["do_sample"] = True

    # Generate the output
    outputs = model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Inference script for GPT2 pretrained model")
    parser.add_argument("--model_name_or_path", type=str, default="gpt2",
                        help="Pretrained model name or path")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Input prompt for text generation")
    parser.add_argument("--decoding_strategy", type=str, choices=["greedy", "beam", "sampling"],
                        default="sampling", help="Decoding strategy to use")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature for text generation (sampling only)")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Maximum number of tokens to generate")
    parser.add_argument("--num_beams", type=int, default=5,
                        help="Number of beams for beam search (only used when decoding_strategy is 'beam')")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model_name_or_path)
    print(f"Using device: {device}")

    generated_text = generate_text(
        model,
        tokenizer,
        args.prompt,
        decoding_strategy=args.decoding_strategy,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device
    )
    print("\n=== Generated Text ===")
    print(generated_text)


if __name__ == "__main__":
    main()

    #python inference.py --model_name_or_path lora_gpt2_medical --prompt "[PATIENT]: What are the symptoms of a migraine? [DOCTOR]:" --decoding_strategy greedy