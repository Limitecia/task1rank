import os
import pandas as pd
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics import f1_score, accuracy_score
from e3_pair_ft.utils_e import apply_span_correction  # 假设已实现该函数

def inference_with_correction(model, tokenizer, prompt, max_length=32, num_beams=5):
    """
    对输入的 prompt 进行编码并生成预测结果，随后调用规则校正函数对结果进行修正。
    返回最终预测的情感标签（已去除特殊标记并转为小写）。
    """
    # 编码输入
    input_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_enc = {k: v.to(model.device) for k, v in input_enc.items()}
    
    # 使用 beam search 生成输出
    outputs = model.generate(
        **input_enc,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    
    # 解码生成文本
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 调用规则校正模块进行后处理
    corrected_output = apply_span_correction(raw_output)
    
    return corrected_output.strip().lower()

def evaluate_inference(model, tokenizer, csv_file, max_length=32, num_beams=5):
    """
    读取验证集 CSV 文件，针对每条数据生成预测结果，
    然后与金标准标签进行比较，计算 F1-measure 和准确率。
    """
    # 读取 CSV 文件，并确保列名无多余空白
    data = pd.read_csv(csv_file, sep="\t", encoding='ISO-8859-1')
    data.columns = data.columns.str.strip()
    
    gold_labels = []
    predictions = []
    
    for idx, row in data.iterrows():
        # 构造包含 Chain-of-Thought 的推理 prompt
        if "context" in row and "source" in row:
            prompt = (
                f"Context: {row['context']}\n"
                f"Source: {row['source']}\n"
                "Step 1: Identify the text span in the source that might cause an emotion.\n"
                "Step 2: Based on commonsense, describe the opinion about that span and why it might evoke emotion.\n"
                "Step 3: Determine the emotion caused towards the last utterance based on the above opinion.\n"
                "Answer:"
            )
        elif "input" in row:
            prompt = row["input"]
        else:
            # 如果缺少构造 prompt 的字段，则跳过
            continue
        
        # 金标准标签：优先使用 "emotion_state"，否则使用 "target"
        if "emotion_state" in row:
            gold = str(row["emotion_state"]).strip().lower()
        elif "target" in row:
            gold = str(row["target"]).strip().lower()
        else:
            continue
        gold_labels.append(gold)
        
        # 生成预测结果，并调用规则校正
        pred = inference_with_correction(model, tokenizer, prompt, max_length=max_length, num_beams=num_beams)
        predictions.append(pred)
        
        print(f"Row {idx}: Gold: {gold}  |  Predicted: {pred}")
    
    # 计算评估指标：weighted F1 和准确率
    f1 = f1_score(gold_labels, predictions, average="weighted")
    accuracy = accuracy_score(gold_labels, predictions)
    
    print("\nEvaluation Results:")
    print(f"Weighted F1 Score: {f1:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return gold_labels, predictions

def main():
    # 模型和 tokenizer 的设置
    model_name = "google/flan-t5-base"
    finetuned_model_dir = "./finetuned_model"  # 与训练时保存路径一致
    
    # 加载 tokenizer 和训练好的模型（如果存在则加载 fine-tuned 模型，否则加载原始模型）
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if os.path.exists(finetuned_model_dir):
        model = T5ForConditionalGeneration.from_pretrained(finetuned_model_dir)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 设置设备，并将模型放到设备上
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 进入评估模式
    
    # 指定验证集 CSV 文件路径
    valid_csv = "data/e3_pair_ft/cause-mult-valid.csv"
    
    # 调用评估函数，对推理结果进行评估
    evaluate_inference(model, tokenizer, valid_csv, max_length=32, num_beams=5)

if __name__ == "__main__":
    main()
