import os
import torch
import pandas as pd
from safetensors.torch import load_file
from transformers import AutoProcessor
# from smolvlm2_video_FT_clean import SmolVLMWithClassifier, get_latest_checkpoint # original
from smolvlm2_video_FT_strategy3 import SmolVLMWithClassifier, get_latest_checkpoint # strategy3
from tqdm import tqdm
import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset inference script")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--epoch", type=str, default=3, required=True, help="fine-tune epoch number")
    parser.add_argument("--model-path", type=str, help="Path to the model directory")
    return parser.parse_args()

args = parse_args()


# 0711
model_path = get_latest_checkpoint(f"C:\\Python_workspace\\TAISC\\code\\smollm-main\\0713_SmolVLM2-500M-Video-Instruct-taisc(strategy3-{args.dataset}-{args.epoch}epoch-complete)") # original
# model_path = "C:\\Python_workspace\\TAISC\\code\\smollm-main\\strategy3 model\\SmolVLM2-500M-Video-Instruct-taisc(latest-cls-strategy3-3epoch)\\checkpoint-36" # strategy3
# model_path = "C:\\Python_workspace\\TAISC\\code\\smollm-main\\SmolVLM2-500M-Video-Instruct-taisc(latest-cls-strategy3-2epoch-complete)-road\\checkpoint-46" # strategy3

model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# 測試用影片與文字（請根據需要替換）
# video_path = "C:\\Python_workspace\\TAISC\\dataset\\freeway_video\\smolvlm2\\train\\freeway_0000.mp4"
# text_prompt = "Is a vehicle collision likely to happen within 1 to 2 seconds?(0: No, 1: Yes)"
# =======================

# 0711
video_folder = f"C:\\Python_workspace\\TAISC\\dataset\\{args.dataset}_video\\smolvlm2\\test"
# video_folder = f"C:\\Python_workspace\\TAISC\\dataset\\{args.dataset}_val_video\\smolvlm2\\val"

# 0711
# output_csv = "{}_infer_results_{}.csv".format(args.dataset , str(datetime.datetime.now())[11:19].replace(":" , "-"))
output_csv = "0713_strategy3_{}_{}epoch_infer_results_{}.csv".format(args.dataset , args.epoch , str(datetime.datetime.now())[11:19].replace(":" , "-"))
text_prompt = "Is a vehicle collision likely to happen within 1 to 2 seconds?(0: No, 1: Yes)"

# === 模型初始化並讀取權重 ===
model = SmolVLMWithClassifier(model_id)
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(state_dict)
model.to("cuda").eval()

# === Processor 載入 ===
processor = AutoProcessor.from_pretrained(model_path)

# 收集影片路徑
video_files = [f for f in os.listdir(video_folder) if f.endswith(".mp4")]
video_paths = [os.path.join(video_folder, f) for f in video_files]

# 儲存推論結果
results = []

for video_path, file_name in tqdm(zip(video_paths, video_files)):
    # 建立對話訊息
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": text_prompt},
            {"type": "video", "path": video_path}
        ]
    }]
    # 處理輸入
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )
    # 依欄位正確轉 dtype
    processed_inputs = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            processed_inputs[k] = v.to("cuda", dtype=model.base_model.dtype)
        elif k in ["input_ids", "attention_mask"]:
            processed_inputs[k] = v.to("cuda", dtype=torch.long)
        else:
            processed_inputs[k] = v.to("cuda")
    # 推論
    with torch.no_grad():
        logits = model(**processed_inputs)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).long().item()

    results.append({"file_name": file_name, "risk": pred})
    tqdm.write(f"[✓] {file_name} → risk: {pred}, logits: {logits[0]:<9.6f}, probs: {probs[0]:<10.8f}")

        
# 儲存結果至 CSV
df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"\n📄 All results saved to: {output_csv}")

# # === 構建對話格式輸入（與訓練時一致） ===
# user_content = [
#     {"type": "text", "text": text_prompt},
#     {"type": "video", "path": video_path}
# ]
# messages = [{"role": "user", "content": user_content}]

# inputs = processor.apply_chat_template(
#     messages,
#     add_generation_prompt=False,
#     tokenize=True,
#     return_tensors="pt",
#     return_dict=True
# )

# # 移至 GPU
# # inputs = {k: v.to("cuda") for k, v in inputs.items() if isinstance(v, torch.Tensor)}
# # inputs = {k: v.to("cuda", dtype=model.base_model.dtype) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
# processed_inputs = {}
# for k, v in inputs.items():
#     if not isinstance(v, torch.Tensor):
#         continue
#     if k == "pixel_values":
#         processed_inputs[k] = v.to("cuda", dtype=model.base_model.dtype)
#     elif k in ["input_ids", "attention_mask"]:
#         processed_inputs[k] = v.to("cuda", dtype=torch.long)
#     else:
#         processed_inputs[k] = v.to("cuda")  # 預設轉 device，但保留原 dtype
# inputs = processed_inputs

# === 推論 ===
# with torch.no_grad():
#     logits = model(**inputs)
#     probs = torch.sigmoid(logits)
#     preds = (probs > 0.5).long()

# === 顯示結果 ===
# print(f"Logits: {logits.item():.4f}")
# print(f"Probability: {probs.item():.4f}")
# print(f"Predicted Label: {preds.item()}")

# === 儲存結果至 CSV ===
# df_output = pd.DataFrame([{
#     "video": os.path.basename(video_path),
#     "logit": logits.item(),
#     "probability": probs.item(),
#     "prediction": preds.item()
# }])
# output_csv_path = "./inference_result.csv"
# df_output.to_csv(output_csv_path, index=False)
# print(f"Inference result saved to: {output_csv_path}")
