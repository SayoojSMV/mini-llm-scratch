# generate.py
import os
import torch
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM
from src.config import LLMConfig

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/instruct_model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Run finetune.py first!")
        return

    # Load Model Structure & Fine-tuned State Dict
    config = LLMConfig()
    model = MiniLLM(config).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)

    tokenizer = Tokenizer()

    print("\n" + "=" * 50)
    print("      Mini-LLM Interactive Generation CLI      ")
    print("==================================================")
    print("Type your prompt and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Exiting generation CLI. Goodbye!")
                break

            # Format input for instruction-following
            formatted_prompt = f"User: {user_input}\nAssistant:"
            input_ids = tokenizer.encode(formatted_prompt).unsqueeze(0).to(device)

            # Generate output tokens
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=60, 
                temperature=0.7, 
                top_k=10
            )

            # Decode output
            generated_text = tokenizer.decode(output_ids[0])

            # Truncate output if the model tries to simulate a new User prompt
            if "\nUser:" in generated_text:
                generated_text = generated_text.split("\nUser:")[0]

            print("\nGenerated Response:")
            print("-" * 40)
            print(generated_text.strip())
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nExiting generation CLI. Goodbye!")
            break

if __name__ == "__main__":
    main()