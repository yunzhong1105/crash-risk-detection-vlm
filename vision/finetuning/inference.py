from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from PIL import Image
import os
import gc
import tqdm

from smolvlm2_video_FT_clean import SmolVLMWithClassifier

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset inference script")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the fine-tune model")
    return parser.parse_args()


def extract_assistant_answer(output_text):
    for line in output_text.strip().split('\n'):
        if line.strip().startswith("Assistant:"):
            return line.replace("Assistant:", "").strip()
    return "99"  # 如果沒有找到

args = parse_args()


# without fine-tune
# model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# with fine-tune
# model_path = "C:\\Python_workspace\\TAISC\\code\\smollm-main\\SmolVLM2-500M-Video-Instruct-taisc(road)\\checkpoint-4450"

# args
model_path = args.model_path

ALL_VIDEO = True
CLS_DECODER = True

processor = AutoProcessor.from_pretrained(model_path)

if CLS_DECODER :
    model = SmolVLMWithClassifier(model_path).to("cuda")
else :
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
    # _attn_implementation="flash_attention_2"
    ).to("cuda")

if not ALL_VIDEO :
    # one image
    image = Image.open("C:\\Python_workspace\\TAISC\\dataset\\freeway\\train\\freeway_0000\\00047.jpg")
    # frame_dir = "C:\\Python_workspace\\TAISC\\dataset\\freeway\\train\\freeway_0000"
    # frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])
    # frames = [Image.open(p).convert("RGB") for p in frame_paths]

    # one sample video
    # video_path = "C:\\Python_workspace\\TAISC\\dataset\\freeway_sample_video\\train\\freeway_0000.mp4"

    # one original video
    video_path = "C:\\Python_workspace\\TAISC\\dataset\\freeway_video\\smolvlm2\\train\\freeway_0000.mp4"

else :
    # all video
    # train
    # video_dir = "C:\\Python_workspace\\TAISC\\dataset\\freeway_video\\smolvlm2\\train"
    
    # test
    # freeway
    # video_dir = "C:\\Python_workspace\\TAISC\\dataset\\freeway_video\\smolvlm2\\miss"
    
    # road
    # video_dir = "C:\\Python_workspace\\TAISC\\dataset\\road_video\\smolvlm2\\test"
    
    # args
    video_dir = f"C:\\Python_workspace\\TAISC\\dataset\\{args.data_dir}_video\\smolvlm2\\test"
    
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    import datetime
    output_csv = "results_{}.csv".format(str(datetime.datetime.now())[11:19].replace(":" , "-"))

'''
messages format
messages = [
    {
        "role": "user",
        "content": [
            # original code
            # {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"},
            
            # image
            # {"type": "image", "image": image},
            
            # video
            {"type": "video", "path": video_path},
            
            # original code
            # {"type": "text", "text": "Can you describe this image?"},
            
            # Q1
            # {"type": "text", "text": "Are there any hazzards in for this dashcam owner?"}
        
            # Q2
            # {"type": "text", "text": "Does this image contain a traffic hazard? Answer: 0 (No hazard), 1 (Hazard)"}
            
            # Q3
            {"type": "text", "text": "Predict whether a car accident will occur within the next 1 to 2 seconds in this video. Answer: 0 (No), 1 (Yes)"}
        ]
    },
]
'''

if ALL_VIDEO :
    df_output = []
    for video_path in tqdm.tqdm(video_paths):
        
        print(f"Processing: {os.path.basename(video_path)}" , end=' : ')
        messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": video_path},
                {"type": "text", "text": "Predict whether a car accident will occur within the next 1 to 2 seconds in this video. Answer: 0 (No), 1 (Yes)"}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

        if not CLS_DECODER :
            with torch.no_grad():
                generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
                
            generated_texts = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )
            
            # print(generated_texts[0])
            df_output.append((os.path.basename(video_path).replace(".mp4" , "") , extract_assistant_answer(generated_texts[0])))
        
        else :
            
            with torch.no_grad():
                logits = model(**inputs)  # shape: [1]
                probs = torch.sigmoid(logits)
                prediction = int((probs > 0.5).long().item())  # 轉為 Python int: 0 或 1

            df_output.append((os.path.basename(video_path).replace(".mp4", ""), prediction))
        
        # release memory
        del inputs
        del generated_ids
        torch.cuda.empty_cache()
        gc.collect()
    
    import pandas as pd
    df = pd.DataFrame(df_output , columns=["file_name", "risk"])
    df.to_csv(output_csv, index=False)
        
else :
    messages = [
    {
        "role": "user",
        "content": [
            # image
            # {"type": "image", "image": image},
            # video
            {"type": "video", "path": video_path},
            {"type": "text", "text": "Predict whether a car accident will occur within the next 1 to 2 seconds in this video. Answer: 0 (No), 1 (Yes)"}
            ]
        },
    ]
    
    inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device, dtype=torch.bfloat16)

    generated_ids = model.generate(**inputs, do_sample=False, max_new_tokens=64)
    generated_texts = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )
    print(generated_texts[0])