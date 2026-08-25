# generate.py
import os
import torch
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM
from src.config import LLMConfig
from src.rag.retriever import RAGRetriever

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_path = "checkpoints/instruct_model.pt"

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file '{checkpoint_path}' not found. Run finetune.py first!")
        return

    # Load Model Structure & State Dict
    config = LLMConfig()
    model = MiniLLM(config).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)

    tokenizer = Tokenizer()
    
    # Initialize RAG Retriever
    retriever = RAGRetriever(knowledge_dir="data/knowledge_base")

    print("\n" + "=" * 50)
    print("      Mini-LLM Interactive RAG Generation CLI      ")
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

            # 1. Retrieve Knowledge Base Context
            context = retriever.retrieve(user_input, top_k=2)
            
            # 2. Format Prompt (Embed context inside User block to match fine-tuning structure)
            if context:
                formatted_prompt = f"User: Based on this information: '{context}', answer: {user_input}\nAssistant:"
                print(f"[RAG Context Injected]: {context}")
            else:
                formatted_prompt = f"User: {user_input}\nAssistant:"

            input_ids = tokenizer.encode(formatted_prompt).unsqueeze(0).to(device)

            # Truncate input if it exceeds max_seq_len
            if input_ids.shape[1] > config.max_seq_len:
                input_ids = input_ids[:, -config.max_seq_len:]

            # 3. Generate Answer (Lower temp for strict context adherence)
            output_ids = model.generate(
                input_ids, 
                max_new_tokens=50, 
                temperature=0.3, 
                top_k=5
            )

            raw_output = tokenizer.decode(output_ids[0])

            # Extract text strictly after "Assistant:"
            if "Assistant:" in raw_output:
                response = raw_output.split("Assistant:")[-1]
            else:
                response = raw_output

            # Truncate if model predicts a new turn
            if "\nUser:" in response:
                response = response.split("\nUser:")[0]

            print("\nGenerated Response:")
            print("-" * 40)
            print(response.strip())
            print("-" * 40 + "\n")

        except KeyboardInterrupt:
            print("\nExiting generation CLI. Goodbye!")
            break

if __name__ == "__main__":
    main()