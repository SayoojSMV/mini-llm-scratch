# app.py
import os
import torch
import gradio as gr
from src.tokenizer import Tokenizer
from src.model.transformer import MiniLLM
from src.config import LLMConfig
from src.rag.retriever import RAGRetriever

# 1. Setup Device & Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoints/instruct_model.pt"

config = LLMConfig()
model = MiniLLM(config).to(device)

if os.path.exists(checkpoint_path):
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        model.load_state_dict(state_dict["state_dict"])
    else:
        model.load_state_dict(state_dict)
    print(f"Loaded checkpoint from {checkpoint_path}")
else:
    print(f"Warning: Checkpoint '{checkpoint_path}' not found! Running un-tuned model.")

model.eval()
tokenizer = Tokenizer()
retriever = RAGRetriever(knowledge_dir="data/knowledge_base")

# 2. Chat Logic Function
def chat_and_retrieve(user_message, history):
    if not user_message.strip():
        return history, ""

    if history is None:
        history = []

    # Retrieve context
    context = retriever.retrieve(user_message, top_k=2)
    
    if context:
        formatted_prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_message}\n\n### Context:\n{context}\n\n### Response:\n"
        display_context = f"**[RAG Context Injected]**: *{context}*"
    else:
        formatted_prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{user_message}\n\n### Response:\n"
        display_context = "*[No RAG Context Injected]*"

    input_ids = tokenizer.encode(formatted_prompt).unsqueeze(0).to(device)
    if input_ids.shape[1] > config.max_seq_len:
        input_ids = input_ids[:, -config.max_seq_len:]

    # Append user turn in dictionary message format
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": ""})

    # Stream Generation with Sampling & Repetition Penalty
    generated_tokens = []
    curr_input = input_ids
    temperature = 0.7
    repetition_penalty = 1.2

    with torch.no_grad():
        for _ in range(80):  # max_new_tokens
            logits, _ = model(curr_input)
            logits = logits[:, -1, :] / temperature

            # Apply repetition penalty to recently generated tokens
            if generated_tokens:
                recent_ids = set([tokenizer.encode(t)[0].item() for t in generated_tokens if len(tokenizer.encode(t)) > 0])
                for token_id in recent_ids:
                    if logits[0, token_id] < 0:
                        logits[0, token_id] *= repetition_penalty
                    else:
                        logits[0, token_id] /= repetition_penalty

            # Top-k sampling (k=40)
            v, _ = torch.topk(logits, 40)
            logits[logits < v[:, [-1]]] = -float('Inf')

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            curr_input = torch.cat((curr_input, next_token), dim=1)
            token_text = tokenizer.decode(next_token[0])
            
            # Stop condition check
            if token_text == "<|endoftext|>" or "\n###" in token_text:
                break

            generated_tokens.append(token_text)
            current_response = "".join(generated_tokens)

            history[-1]["content"] = current_response.strip()
            yield history, display_context

# 3. File Upload Handler
def upload_file(file):
    if file is None:
        return "No file uploaded."
    
    os.makedirs("data/knowledge_base", exist_ok=True)
    filename = os.path.basename(file.name)
    destination = os.path.join("data/knowledge_base", filename)

    with open(file.name, "r", encoding="utf-8", errors="ignore") as src_f:
        content = src_f.read()

    with open(destination, "w", encoding="utf-8") as dst_f:
        dst_f.write(content)

    retriever.load_and_index_documents()
    return f"Successfully indexed `{filename}` into Knowledge Base!"

# 4. Gradio UI Layout
with gr.Blocks(title="Mini-LLM Scratch Studio") as demo:
    gr.Markdown("# 🤖 Mini-LLM Interactive RAG Studio")
    gr.Markdown("A GPT-style language model built from scratch with real-time vector retrieval.")

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=450)
            msg = gr.Textbox(placeholder="Ask Mini-LLM a question...", show_label=False)
            clear = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            gr.Markdown("### 📚 Knowledge Base Manager")
            context_box = gr.Markdown("*RAG status and retrieved snippets will appear here.*")
            file_uploader = gr.File(label="Upload .txt File to RAG Storage", file_types=[".txt"])
            upload_status = gr.Markdown()

    msg.submit(chat_and_retrieve, [msg, chatbot], [chatbot, context_box])
    msg.submit(lambda: "", None, msg)
    clear.click(lambda: None, None, chatbot, queue=False)
    file_uploader.change(upload_file, inputs=[file_uploader], outputs=[upload_status])

if __name__ == "__main__":
    demo.queue().launch(server_name="127.0.0.1", server_port=7860)