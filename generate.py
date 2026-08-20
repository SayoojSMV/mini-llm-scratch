# generate.py
import os
import torch
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Run train.py first!")
        return

    # Load Model & Tokenizer
    model = MiniLLM.load_checkpoint(checkpoint_path, device=device)
    tokenizer = Tokenizer()

    print("\n" + "=" * 50)
    print("      Mini-LLM Interactive Generation CLI      ")
    print("=" * 50)
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            prompt = input("Prompt > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ["exit", "quit"]:
                print("Exiting generation CLI. Goodbye!")
                break

            # Encode input
            input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)

            # Generate output tokens
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=40, 
                temperature=0.8, 
                top_k=10
            )

            # Decode and print output
            generated_text = tokenizer.decode(output_ids[0])
            print("\nGenerated Response:")
            print("-" * 40)
            print(generated_text)
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nExiting generation CLI. Goodbye!")
            break

if __name__ == "__main__":
    main()