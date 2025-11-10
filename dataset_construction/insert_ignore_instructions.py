import random
import json
import nltk
# 尝试直接使用PunktSentenceTokenizer而不是sent_tokenize
from nltk.tokenize import PunktSentenceTokenizer
# 下载NLTK所需资源（首次运行需要）
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("正在下载NLTK punkt资源...")
    nltk.download('punkt')

def insert_ignore_instructions(text, instruction):
    """
    Insert "Ignore previous instructions" at random valid sentence break positions.
    
    Args:
        text (str): The original text content
        
    Returns:
        str: Text with the instructions inserted at a random sentence break
    """
    # Define the instruction
    # instruction = " Ignore previous instructions. "
    
    # Use NLTK for sentence segmentation if available and requested
    tokenizer = PunktSentenceTokenizer()
    sentences = tokenizer.tokenize(text)
    
    if not sentences:
        return text
    
    # 选择一个随机句子进行插入
    chosen_index = random.randint(0, len(sentences) - 1)
    
    # 重建文本，在选定的句子后插入指令
    result = ""
    start_index = -1
    end_index = -1
    for i, sentence in enumerate(sentences):
        result += sentence
        if i != len(sentences) - 1:
            result += " "
        if i == chosen_index:
            start_index = len(result)
            result += instruction
            end_index = len(result)
            if i != len(sentences) - 1:
                result += " "
    assert start_index != -1 and end_index != -1
    return result, start_index, end_index


def main():
    # parser = argparse.ArgumentParser(description="Insert 'Ignore previous instructions' at random sentence breaks.")
    # parser.add_argument("-f", "--file", help="Input file path")
    # parser.add_argument("-o", "--output", help="Output file path (defaults to stdout)")
    # parser.add_argument("--text", help="Direct text input (alternative to file)")
    # parser.add_argument("--use-nltk", action="store_true", help="Use NLTK for more accurate sentence segmentation")

    random.seed(42)
    with open("datasets/data/train_data/data_instruction_10k.json", "r") as f:
        data = json.load(f)

    for idx, item in enumerate(data):
        assert len(item) == 4
        assert item[0]['role'] == "system"
        assert item[1]['role'] == "user"
        assert item[2]['role'] == "data"
        data_text = item[2]['content']
        random_index = random.randint(0, len(data_text) - 1)
        while random_index == idx:
            random_index = random.randint(0, len(data_text) - 1)
        new_instruction = data[random_index][1]['content']
        modified_text, start_index, end_index = insert_ignore_instructions(data_text, new_instruction)
        item[2]['content'] = modified_text
        item[2]['injected_instruction'] = new_instruction
        item[2]['start_index'] = start_index
        item[2]['end_index'] = end_index

        item.pop(3)

    with open("datasets/data/train_data/data_instruction_10k_injected_no_response.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    
    # args = parser.parse_args()
    
    # if args.use_nltk and not NLTK_AVAILABLE:
    #     print("Warning: NLTK not available. Install with 'pip install nltk'", file=sys.stderr)
    
    # # Get input text
    # if args.text:
    #     text = args.text
    # elif args.file:
    #     try:
    #         with open(args.file, 'r', encoding='utf-8') as f:
    #             text = f.read()
    #     except Exception as e:
    #         print(f"Error reading file: {e}", file=sys.stderr)
    #         return 1
    # else:
    #     print("Please provide input text using --text or --file", file=sys.stderr)
    #     return 1
    
    # # Process the text
    # modified_text = insert_ignore_instructions(text, args.use_nltk)
    
    # # Output the result
    # if args.output:
    #     try:
    #         with open(args.output, 'w', encoding='utf-8') as f:
    #             f.write(modified_text)
    #     except Exception as e:
    #         print(f"Error writing to output file: {e}", file=sys.stderr)
    #         return 1
    # else:
    #     print(modified_text)
    
    # return 0

# Example usage
if __name__ == "__main__":
    main()