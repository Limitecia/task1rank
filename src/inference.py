import os
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
# 假设 utils_e.py 中已经实现了 apply_span_correction 函数
from e3_pair_ft.utils_e import apply_span_correction

def inference_with_correction(model, tokenizer, prompt, max_length=32, num_beams=5):
    """
    对输入的 prompt 进行编码，然后使用 beam search 生成预测结果，
    再调用 apply_span_correction 对生成结果进行规则校正。
    """
    # 编码输入 prompt
    input_enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_enc = {k: v.to(model.device) for k, v in input_enc.items()}
    
    # 使用 beam search 生成输出
    outputs = model.generate(
        **input_enc,
        max_length=max_length,
        num_beams=num_beams,
        early_stopping=True
    )
    
    # 解码生成的输出文本，去除特殊标记
    raw_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("Raw Output:", raw_output)
    
    # 调用规则校正模块对生成文本进行后处理
    corrected_output = apply_span_correction(raw_output)
    return corrected_output

def main():
    # 模型和 tokenizer 的名称或路径
    model_name = "google/flan-t5-base"
    # 训练好的模型存放目录，需与训练时 output_dir 保持一致
    finetuned_model_dir = "./finetuned_model"
    
    # 加载 tokenizer 和模型（从 fine-tuned 模型目录加载优先）
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    if os.path.exists(finetuned_model_dir):
        model = T5ForConditionalGeneration.from_pretrained(finetuned_model_dir)
    else:
        model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # 设置设备（GPU/CPU）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # 设置为评估模式
    
    # 构造包含 Chain-of-Thought 多步推理的 prompt
    prompt = (
        "Context: Joey: Hey, how are you?\n"
        "Source: Chandler: I'm feeling a bit down today.\n"
        "Step 1: Identify the text span in the source that might cause an emotion.\n"
        "Step 2: Based on commonsense, describe the opinion about that span and why it might evoke emotion.\n"
        "Step 3: Determine the emotion caused towards the last utterance based on the above opinion.\n"
        "Answer:"
    )
    
    # 调用推理函数生成预测结果，并进行后处理
    final_output = inference_with_correction(model, tokenizer, prompt, max_length=32, num_beams=5)
    print("Final Corrected Output:", final_output)

if __name__ == "__main__":
    main()
