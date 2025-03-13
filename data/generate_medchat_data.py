import os
from datasets import load_dataset

def main():
    # 1. Load the dataset.
    #    The dataset has only a 'train' split (based on the HF page),
    #    so we'll load split="train".
    dataset = load_dataset("ruslanmv/ai-medical-chatbot", split="train")

    # 2. Prepare output file
    output_file = "medical_chat_processed.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        # 3. Iterate over each example and write in the desired format
        for example in dataset:
            patient_text = example.get("Patient", "")
            doctor_text = example.get("Doctor", "")

            # Format: [PATIENT]: <patient> [DOCTOR]: <doctor>
            line = f"[PATIENT]: {patient_text} [DOCTOR]: {doctor_text}\n"
            f.write(line)

    print(f"Processed dataset saved to {output_file}")


if __name__ == "__main__":
    main()