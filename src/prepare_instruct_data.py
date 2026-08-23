# src/prepare_instruct_data.py
import json
import os

def format_alpaca():
    raw_path = os.path.join("data", "alpaca_raw.json")
    out_path = os.path.join("data", "instruct_data.txt")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Missing {raw_path}. Ensure the JSON file is present.")

    with open(raw_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    formatted_text = ""
    for entry in data[:3000]:
        instruction = entry.get("instruction", "")
        input_text = entry.get("input", "")
        output = entry.get("output", "")

        if input_text:
            prompt = f"User: {instruction}\nContext: {input_text}\nAssistant: {output}\n\n"
        else:
            prompt = f"User: {instruction}\nAssistant: {output}\n\n"

        formatted_text += prompt

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted_text)

    print(f"Successfully generated {out_path}!")

if __name__ == "__main__":
    format_alpaca()