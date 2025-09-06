import numpy as np
import torch
import math
import pandas as pd
import os
import torch.nn as nn
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel

### Preprocessing of .txt files

def read_and_format_txt_files(folder_path):
    """
    Reads all txt files from the given folder path, formats the content into prompts,
    and stores them in a dictionary.

    Args:
    folder_path (str): The path to the folder containing the txt files.

    Returns:
    dict: A dictionary where keys are filenames and values are lists of formatted prompts.
    """
    prompts_dict = {}

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()

            sections = content.split('\n\n') 
            formatted_prompts = []

            for section in sections:
                if ':' in section:
                    title, sentences = section.split(':', 1) # title: 小标题
                    title = title.strip()
                    sentences = sentences.strip().split('. ') # 按句让模型学习
                    for sentence in sentences:
                        if sentence:  # 确保非空
                            prompt = f"Please carefully read and remember the following facts about Beijing National Day School's {title} in {filename[:-4]}: "
                            prompt_length = len(prompt.split(" ")) # 记录提示词长度，在Dataset里传给attention mask，之后在训练的时候避免模型学习提示词
                            knowledege = prompt + sentence + "."
                            prompt = prompt.replace("\n","")
                            formatted_prompts.append((knowledege,prompt_length)) # 以元组储存
                else:
                    sentences = section.strip().split('. ')
                    for sentence in sentences:
                        if sentence:  # Ensure the sentence is not empty
                            prompt = f"Please carefully read and remember the following facts about Beijing National Day School's {filename[:-4]}: "
                            prompt_length = len(prompt.split(" "))
                            knowledege = prompt + sentence + "."
                            prompt = prompt.replace("\n","")
                            formatted_prompts.append((knowledege,prompt_length))

            prompts_dict[filename[:-4]] = formatted_prompts

    return prompts_dict

### Loading and Training Model

class KnowledgeDataset(Dataset):
    def __init__(self, prompt_dic, tokenizer, max_length=128):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        for _, knowledge_list in prompt_dic.items():
            for (sentence, prompt_length) in knowledge_list:
                encoding = tokenizer(
                    sentence,
                    max_length=max_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

                attention_mask = encoding['attention_mask'].flatten()
                attention_mask[:prompt_length] = 0 # 把提示词部分的掩码设为0
                
                self.examples.append({
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': attention_mask,
                    'labels': encoding['input_ids'].flatten()
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]

def train(model, dataloader, optimizer, num_epochs, device):

    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        
        epoch_loss = 0
        
        for batch_idx, batch in enumerate(dataloader):

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            epoch_loss += loss.item()
            
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
        if epoch % 2 == 1:
            print(f"Epoch {epoch+1}/{num_epochs} Loss: {epoch_loss:.4f}")

### Model Evaluation

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

def model_evaluation(model, df:pd.DataFrame):
    """
    获取模型表现，由F1-score，目标-预测相似度，提问-预测相关性构成。

    参数：
    model: 模型，用来计算参数量
    df: 三列的DataFrame，第一列"prompt"，第二列"truth"，第三列"prediction"

    输出：
    score: 模型在数据上的平均得分
    """

    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')

    def normalize_text(text):
        """文本标准化处理"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)  # 移除标点
        text = re.sub(r'\s+', ' ', text).strip()  # 标准化空格
        return text

    def exact_match(pred:str, true:str):
        """精确匹配"""
        pred_norm = normalize_text(pred)
        true_norm = normalize_text(true)
        return pred_norm == true_norm

    def f1_score(pred:str, true:str):
        """F1分数"""
        pred_tokens = set(normalize_text(pred).split())
        true_tokens = set(normalize_text(true).split())
        
        if not pred_tokens or not true_tokens:
            return 0.0
        
        common_tokens = pred_tokens & true_tokens
        precision = len(common_tokens) / len(pred_tokens)
        recall = len(common_tokens) / len(true_tokens)
        
        if precision + recall == 0:
            return 0.0
            
        f1 = 2 * (precision * recall) / (precision + recall)
        return f1

    
    sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
    
    def truth_prediciton_similarity(pred:str, true:str):
        """目标与预测之间的语义相似度"""
        pred_emb = sentence_transformer.encode([pred], convert_to_tensor=True)
        true_emb = sentence_transformer.encode([true], convert_to_tensor=True)
        cosine_sim = cosine_similarity(pred_emb, true_emb).item()
        return cosine_sim


    vectorizer = TfidfVectorizer()
    all_texts = df["prompt"].tolist() + df["prediction"].tolist()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    def prompt_prediciton_relavance(prompt:str,pred:str):
        """问题与预测之间的相关性"""
        question_vector = vectorizer.transform([prompt])
        answer_vector = vectorizer.transform([pred])
        cosine_sim = cosine_similarity(question_vector, answer_vector)[0][0]
        return cosine_sim

    def model_parameter_score(model):
        """模型参数量评分"""
        paramcount = model.paramcount
        threshold = 1e9
        if paramcount > threshold:
            return -1
        else:
            return math.log(threshold - paramcount) / 9

    total_score
    for prompt, truth, pred in zip(df['prompt'],df['truth'],df['prediction']):
        if exact_match(pred, truth):
            total_score += 1
        else:
            F1 = f1_score(pred,truth)
            TPsim = truth_prediciton_similarity(pred,truth)
            PPrel = prompt_prediciton_relavance(pred,truth)
            mps = model_parameter_score(model)
            total_score += mps * ( F1 + TPsim + PPrel ) / 3

    return total_score / len(df)

### Main

def main():

    folder_path = '/personal/知识库 - 格式化txt/' 
    prompts_dict = read_and_format_txt_files(folder_path)
    
    for i, j in prompts_dict.items():
        print(i, len(j))

    # Initialize tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('/personal/gpt2/')
    model = GPT2LMHeadModel.from_pretrained('/personal/gpt2/')

    # Add padding token if not present (GPT2 doesn't have default padding)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'}) 
        model.resize_token_embeddings(len(tokenizer))

    # Pre-training example
    with torch.no_grad():
        encoding = tokenizer(
            "Tell me about Beijing National Day School's student housing conditions.",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    temperature = 0.7 
    top_k = 50  # Select from top 50 candidate words

    outputs = model.generate(
        input_ids=encoding["input_ids"],
        attention_mask=encoding["attention_mask"],
        max_new_tokens=100,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("Generated text before training:")
    print(generated_text)

    # Prepare Training
    dataset = KnowledgeDataset(prompts_dict, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    train(model, dataloader, optimizer, 10, device='cuda')

    # Post-training experiment
    with torch.no_grad():
        encoding = tokenizer(
            "Beijing National Day School's theatre is located ",
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    temperature = 0.7 
    top_k = 100

    outputs = model.generate(
        input_ids=encoding["input_ids"].to('cuda'),
        attention_mask=encoding["attention_mask"].to('cuda'),
        max_new_tokens=80,
        temperature=temperature,
        top_k=top_k,
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True
    )

    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("\nGenerated text after training:")
    print(generated_text)

if __name__ == "__main__":
    main()
