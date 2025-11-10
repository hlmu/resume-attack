import json
import argparse
import os

from crawl_responses import clean_conv, create_classification_prompt

def parse_args():
    parser = argparse.ArgumentParser(description='Copy data to LLaMA-Factory')
    parser.add_argument('--input', default='datasets/data/train_data/data_instruction_10k_injected_Qwen_Qwen3-8B.json',
                        help='Input file')
    parser.add_argument('--output', default='LLaMA-Factory/data/data_instruction_10k_injected_Qwen_Qwen3-8B_lf.json',
                        help='Output file')
    return parser.parse_args()

def main():
    # print working directory
    print(os.getcwd())
    args = parse_args()
    with open(args.input, "r") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        conv_item_cleaned = clean_conv(item)
        item_new = create_classification_prompt(conv_item_cleaned, "", "You are a helpful assistant.")
        new_data.append({
            "messages": item_new,
        })

    with open(args.output, "w") as f:
        json.dump(new_data, f, indent=4)

    with open("LLaMA-Factory/data/dataset_info.json", "r") as f:
        dataset_info = json.load(f)

    dataset_name = args.output.split("/")[-1].split(".")[0]

    dataset_info[dataset_name] = {
        "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "system_tag": "system"
        }
    }

    with open("LLaMA-Factory/data/dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

if __name__ == "__main__":
    main()