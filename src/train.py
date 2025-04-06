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

class EmotionCauseDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_input_length=512, max_output_length=32):
        # 读取 CSV 文件时指定制表符分隔、跳过格式错误的行，并指定编码
        self.data = pd.read_csv(csv_file, sep="\t", on_bad_lines='skip', encoding='ISO-8859-1')
        # 对列名去除多余空白，确保字段名称准确
        self.data.columns = self.data.columns.str.strip()
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
            target_text = str(row["emotion_state"])
        elif "target" in row:
            target_text = str(row["target"])
        else:
            print("CSV columns:", list(self.data.columns))
            raise KeyError("CSV文件中缺少目标文本字段，例如 'emotion_state' 或 'target'。")
        
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
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def compute_metrics(pred):
    """
    计算 F1-measure 指标，假设目标标签为单词，先将模型生成的 token 序列解码为文本字符串，
    然后与参考答案比较。这里采用 weighted F1 作为示例指标。
    """
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    predictions, labels = pred
    # 解码预测文本
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # 将 label 中 -100 替换为 pad_token_id，再解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # 将预测与真实文本转换为列表（假设为单词标签），并转为小写
    decoded_preds = [pred.strip().lower() for pred in decoded_preds]
    decoded_labels = [label.strip().lower() for label in decoded_labels]
    
    # 计算 weighted F1
    f1 = f1_score(decoded_labels, decoded_preds, average='weighted')
    return {"f1": f1}

def main():
    # 模型及输出目录设置
    model_name = "google/flan-t5-base"  # 使用 Flan-T5-base 模型（250M参数）
    output_dir = "./finetuned_model"
    # CSV 文件路径（预处理阶段生成的文件）
    train_csv = "data/e3_pair_ft/cause-mult-train.csv"
    valid_csv = "data/e3_pair_ft/cause-mult-valid.csv"
    
    # 训练超参数设置，参照论文：
    num_train_epochs = 3            # 训练周期：2-3个 epoch
    per_device_train_batch_size = 32  # 每个设备批次大小 16
    per_device_eval_batch_size = 32

    # 加载 tokenizer 和模型
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 构造训练和验证数据集
    train_dataset = EmotionCauseDataset(train_csv, tokenizer)
    valid_dataset = EmotionCauseDataset(valid_csv, tokenizer)
    
    # 设置训练参数，同时增加早停和加载最佳模型的配置，并指定 TensorBoard 日志记录
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
        load_best_model_at_end=True,      # 在训练结束时加载验证集上表现最好的模型
        metric_for_best_model="f1",         # 以 F1 指标作为最佳模型评价依据
        report_to=["tensorboard"]         # 指定使用 TensorBoard 记录日志
    )
    
    # 构造 Trainer，并添加早停策略回调
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
