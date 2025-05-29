# ==============================================线上模型=================================

from openai import OpenAI # type: ignore
import os
from dotenv import load_dotenv # type: ignore
import tiktoken
load_dotenv()

# client = OpenAI(
#     api_key=os.getenv("QWEN_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
# )

# client = OpenAI(api_key=os.getenv("DeepSeek_API_KEY"), base_url="https://api.deepseek.com")

# 获取模型的编码器（用于计算 tokens）
def count_tokens(prompt):
    enc = tiktoken.get_encoding("cl100k_base")  # 使用 GPT-2 的编码器，适用于大部分 GPT 模型
    tokens = enc.encode(prompt)
    return len(tokens)

# 定义函数来调用 ChatGPT
def ask_llm_online(model_type):
    if model_type.startswith("qwen"):
        client = OpenAI(api_key=os.getenv("QWEN_API_KEY"), base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    elif model_type.startswith("deepseek"):
        client = OpenAI(api_key=os.getenv("DeepSeek_API_KEY"), base_url="https://api.deepseek.com")
    elif model_type.startswith("moonshot"):
        client = OpenAI(api_key=os.getenv("MoonShot_API_KEY"), base_url="https://api.moonshot.cn/v1")

    with open("output_with_text.txt", "r", encoding="utf-8") as file:
        prompt = file.read()

    # 计算输入 token 的数量
    input_token_count = count_tokens(prompt)
    print(f"输入token数量: {input_token_count}")

    response = client.chat.completions.create(
        model= model_type,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            # {
            #     "role": "system",
            #     "content": "You are an expert in data science and machine learning. Your role is to analyze high-dimensional data and uncover hidden patterns or correlations. Focus on tasks such as identifying clusters, key features, or trends in the data. If necessary, propose dimensionality reduction techniques (e.g., PCA, t-SNE, UMAP) or clustering methods (e.g., K-means, DBSCAN). Explain your findings in a way that is interpretable to someone with a basic understanding of data science."
            # },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,  # 设置温度较低以提高回答的一致性
        max_tokens= 100 #8192  # 设置回答的最大 token 数量
    )
    # 获取并返回模型生成的文本
    return response.choices[0].message.content

# ==============================================本地模型=================================
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from transformers import AutoModelForCausalLM, AutoTokenizer

# 全局变量，缓存模型和 Tokenizer
_loaded_model = None
_loaded_tokenizer = None

def ask_llm_offline(model_type):
    global _loaded_model, _loaded_tokenizer

    # 如果模型和 Tokenizer 尚未加载，则加载
    if _loaded_model is None or _loaded_tokenizer is None:
        _loaded_model = AutoModelForCausalLM.from_pretrained(
            model_type,
            torch_dtype="auto",
            device_map="auto",
            # force_download=True
        )
        _loaded_tokenizer = AutoTokenizer.from_pretrained(model_type)
        print("Model and tokenizer loaded.")

    # 使用已加载的模型和 Tokenizer
    model = _loaded_model
    tokenizer = _loaded_tokenizer

    with open("output_with_text.txt", "r", encoding="utf-8") as file:
        prompt = file.read()

    messages = [
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
        temperature=0.2  # 控制生成的随机性
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response

# Qwen/Qwen2.5-7B-Instruct qwen2.5-7b-instruct Meta-Llama-3-8B-Instruct
# ask_llm("Qwen/Qwen2.5-7B-Instruct")
