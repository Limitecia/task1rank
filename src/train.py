import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import f1_score
import numpy as np

# Debug 模式标志：True 时只加载部分数据，False 时加载全部数据
DEBUG_MODE = True

class EmotionCauseDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_input_length=256, max_output_length=32, limit=None):
        # 读取 CSV 文件时指定制表符分隔、跳过格式错误的行，并指定编码
        self.data = pd.read_csv(csv_file, sep="\t", on_bad_lines='skip', encoding='ISO-8859-1')
        # 对列名去除多余空白，确保字段名称准确
        self.data.columns = self.data.columns.str.strip()
        # 如果设置了 limit，则只取前 limit 行
        if limit is not None:
            self.data = self.data.head(limit)
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        # CSV 文件应至少包含 "context", "source", "emotion_state" 字段

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # 如果 CSV 中包含额外信息（如 c_id, u1, u2），则拼接到 prompt 中
        extra_info = ""
        for field in ["c_id", "u1", "u2"]:
            if field in row:
                extra_info += f"{field}: {row[field]}\n"
        
        # 构造包含 Chain-of-Thought 的 prompt 模板
        if "context" in row and "source" in row:
            prompt = (
                f"Context: {row['context']}\n"
                f"Source: {row['source']}\n"
                f"{extra_info}" +
                "Step 1: Identify the text span in the source that might cause an emotion.\n"
                "Step 2: Based on commonsense, describe the opinion about that span and why it might evoke emotion.\n"
                "Step 3: Determine the emotion caused towards the last utterance based on the above opinion.\n"
                "Answer:"
            )
        elif "input" in row:
            prompt = row["input"]
        else:
            print("CSV columns:", list(self.data.columns))
            raise KeyError("CSV文件中缺少用于构造 prompt 的字段，例如 'context' 和 'source' 或 'input'。")
        
        # 目标文本：优先使用 "emotion_state"，否则尝试 "target"
        if "emotion_state" in row:
            target_text = str(row["emotion_state"]).strip()
        elif "target" in row:
            target_text = str(row["target"]).strip()
        else:
            print("CSV columns:", list(self.data.columns))
            raise KeyError("CSV文件中缺少目标文本字段，例如 'emotion_state' 或 'target'。")
        
        # 如果目标文本为空，则使用默认值 "neutral"
        if target_text == "":
            print(f"Warning: Empty target text at index {idx}, setting to 'neutral'.")
            target_text = "neutral"
        
        # 对 prompt 和目标文本分别进行编码
        input_encodings = self.tokenizer(
            prompt,
            max_length=self.max_input_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        target_encodings = self.tokenizer(
            target_text,
            max_length=self.max_output_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        input_ids = input_encodings.input_ids.squeeze()
        attention_mask = input_encodings.attention_mask.squeeze()
        labels = target_encodings.input_ids.squeeze()
        # 将填充位置标记为 -100，方便计算 loss 时忽略
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        # 检查如果所有标签都为 -100，则说明目标文本未能生成有效token
        if torch.all(labels == -100):
            print(f"Warning: Target text for index {idx} resulted in all pad tokens. Using default label 'neutral'.")
            default_tokens = self.tokenizer("neutral", max_length=self.max_output_length,
                                            padding="max_length", truncation=True, return_tensors="pt").input_ids.squeeze()
            default_tokens[default_tokens == self.tokenizer.pad_token_id] = -100
            labels = default_tokens
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def compute_metrics(pred):
    """
    计算 F1-measure 指标，假设目标标签为单词，
    解码预测和真实标签后计算 weighted F1 分数。
    """
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    predictions, labels = pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]
    
    f1 = f1_score(decoded_labels, decoded_preds, average='weighted')
    return {"f1": f1}

def main():
    # 模型及输出目录设置
    model_name = "google/flan-t5-base"
    output_dir = "./finetuned_model"
    # CSV 文件路径（预处理阶段生成的文件）
    train_csv = "data/e3_pair_ft/cause-mult-train.csv"
    valid_csv = "data/e3_pair_ft/cause-mult-valid.csv"
    
    # 训练超参数设置
    num_train_epochs = 3
    per_device_train_batch_size = 8  # 调试模式下使用较小批次
    per_device_eval_batch_size = 8

    # 如果在调试模式下，则只加载部分数据
    train_limit = 100 if DEBUG_MODE else None
    valid_limit = 20 if DEBUG_MODE else None

    # 加载 tokenizer 和模型
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 构造训练和验证数据集
    train_dataset = EmotionCauseDataset(train_csv, tokenizer, limit=train_limit)
    valid_dataset = EmotionCauseDataset(valid_csv, tokenizer, limit=valid_limit)
    
    # 调试输出：打印第一个样本信息
    sample = train_dataset[0]
    print("Sample input_ids:", sample["input_ids"][:10])
    print("Sample labels:", sample["labels"][:10])
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=2e-4,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        save_total_limit=1,
        fp16=True if torch.cuda.is_available() else False,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to=["tensorboard"]
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)]
    )
    
    trainer.train()
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    main()
