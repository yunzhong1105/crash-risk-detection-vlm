import torch
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from transformers import AutoProcessor, BitsAndBytesConfig, AutoModelForImageTextToText
import os
import re
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import argparse

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset inference script")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset directory")
    parser.add_argument("--epoch", type=int, default=3, required=True, help="fine-tune epoch number")
    return parser.parse_args()

##########----------SmolVLMWithClassifier version 1----------##########

class SmolVLMWithClassifier(nn.Module):
    def __init__(self, model_id):
        super().__init__()
        self.base_model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        
        # 部分參數凍結策略：只微調 text decoder + classifier
        for name, param in self.base_model.named_parameters():
            # print(name)
            
            # if "vision_model" in name:
            #     param.requires_grad = False  # 凍結 vision encoder
            
            if "text_model" in name:
                param.requires_grad = False  # 凍結 text encoder
        
        # assert False
        
        H = self.base_model.config.text_config.hidden_size
        # self.classifier = nn.Linear(H, 1).to(torch.bfloat16)  # binary classification
        self.classifier = nn.Sequential(
            nn.Linear(H, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        ).to(torch.bfloat16)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, **kwargs):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = output.hidden_states[-1]  # shape: [B, T, H]
        cls_token = last_hidden[:, -1, :]  # shape: [B, H]
        logits = self.classifier(cls_token)  # shape: [B, 1]
        return logits.squeeze(-1)

##########----------BCEWithLogitsLoss----------##########

class BinaryClassificationTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("classification_labels").float().to(model.base_model.device)
        logits = model(**inputs)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels)
        return (loss, logits) if return_outputs else loss

def build_collate_fn(video_base_path, processor, image_token_id):
    def collate_fn(examples):
        instances = []

        for example in examples:
            prompt = example["text"]
            if len(prompt) > 200:
                prompt = prompt[:200]

            # split train(50/50) to try model
            # video_path = os.path.join(f"C:\\Python_workspace\\TAISC\\dataset\\{video_base_path}_sample_val_video\\smolvlm2\\train", example["video"])
            video_path = os.path.join(f"C:\\Python_workspace\\TAISC\\dataset\\{video_base_path}_sample_video\\smolvlm2\\train", example["video"])

            user_content = [
                {"type": "text", "text": prompt},
                {"type": "video", "path": video_path}
            ]

            messages = [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": [{"type": "text", "text": example["label"]}]}
            ]

            instance = processor.apply_chat_template(
                messages,
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                truncation=False
            )
            
            instance["label"] = torch.tensor(example["label"], dtype=torch.long)
            instances.append(instance)

        input_ids = pad_sequence(
            [inst["input_ids"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=processor.tokenizer.pad_token_id
        )

        attention_mask = pad_sequence(
            [inst["attention_mask"].squeeze(0) for inst in instances],
            batch_first=True,
            padding_value=0
        )

        labels = pad_sequence(
            [inst["input_ids"].squeeze(0).clone() for inst in instances],
            batch_first=True,
            padding_value=-100
        )
        labels[labels == image_token_id] = -100

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "classification_labels": torch.stack([inst["label"] for inst in instances])
        }

        pvs = [inst["pixel_values"].squeeze(0) for inst in instances if "pixel_values" in inst]
        if pvs:
            max_frames = max(pv.shape[0] for pv in pvs)
            max_h = max(pv.shape[-2] for pv in pvs)
            max_w = max(pv.shape[-1] for pv in pvs)
        else:
            max_h = max_w = processor.video_size['longest_edge']
            max_frames = 1

        padded_pixel_values_list = []
        for ex in instances:
            pv = ex.get("pixel_values", None)
            if pv is None:
                shape_pv = (max_frames, 3, max_h, max_w)
                padded_pv = torch.zeros(shape_pv, dtype=torch.float32)
            else:
                pv = pv.squeeze(0)
                f, c, h, w = pv.shape
                padded_pv = torch.zeros(
                    (max_frames, c, max_h, max_w),
                    dtype=pv.dtype,
                    device=pv.device
                )
                padded_pv[:f, :, :h, :w] = pv
            padded_pixel_values_list.append(padded_pv)

        out["pixel_values"] = torch.stack(padded_pixel_values_list, dim=0)
        return out

    return collate_fn

def get_latest_checkpoint(output_dir):
    checkpoints = [
        d for d in os.listdir(output_dir)
        if re.match(r"^checkpoint-\d+$", d) and os.path.isdir(os.path.join(output_dir, d))
    ]
    if not checkpoints:
        return None
    # 依據數字排序，取出最大值
    latest = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    return os.path.join(output_dir, latest)


# for QLoRA
import sys
# 強制將 bitsandbytes-windows 當成 bitsandbytes
try:
    import bitsandbytes as bnb
    sys.modules["bitsandbytes"] = bnb
except ImportError:
    raise ImportError("請先安裝 bitsandbytes-windows：pip install bitsandbytes-windows")

if __name__ == "__main__" :
    
    USE_LORA = False
    USE_QLORA = False
    SMOL = True
    
    args = parse_args()

    model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if SMOL else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

    processor = AutoProcessor.from_pretrained(
        model_id,
    )

    if USE_QLORA or USE_LORA:
        lora_config = LoraConfig(
            r=8,
            lora_alpha=8,
            lora_dropout=0.1,
            target_modules=['down_proj','o_proj','k_proj','q_proj','gate_proj','up_proj','v_proj'],
            use_dora=False if USE_QLORA else True,
            init_lora_weights="gaussian"
        )
        lora_config.inference_mode = False
        if USE_QLORA:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config if USE_QLORA else None,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )
        # model.add_adapter(lora_config)
        # model.enable_adapters()
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        print(model.get_nb_trainable_parameters())
    else:
        model = SmolVLMWithClassifier(model_id).to("cuda")
        
        # model = AutoModelForImageTextToText.from_pretrained(
        #     model_id,
        #     torch_dtype=torch.bfloat16,
        #     # _attn_implementation="flash_attention_2",
        # ).to("cuda")

        # if you'd like to only fine-tune LLM
        # for param in model.model.vision_model.parameters():
        #     param.requires_grad = False

    peak_mem = torch.cuda.max_memory_allocated()
    print(f"The model as is is holding: {peak_mem / 1024**3:.2f} of GPU RAM")

    smolvlm2 = load_dataset(
        path = f"C:\\Python_workspace\\TAISC\\dataset\\{args.dataset}_sample_video\\smolvlm2\\dataset_script.py" , 
        name = "smolvlm2" , 
        data_dir = f"C:\\Python_workspace\\TAISC\\dataset\\{args.dataset}_sample_video\\smolvlm2" , 
        trust_remote_code = True
    )
   
    train_smolvlm2 = smolvlm2["train"]

    # split_smolvlm2 = smolvlm2["train"].train_test_split(test_size=0)
    # train_smolvlm2 = split_smolvlm2["train"]
    # 0710
    # eval_smolvlm2 = split_smolvlm2["test"]

    # del split_smolvlm2, smolvlm2

    image_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<image>")
    ]


    model_name = model_id.split("/")[-1]

    collate_fn = build_collate_fn(
        video_base_path=args.dataset,
        processor=processor,
        image_token_id=image_token_id
    )
    
    training_args = TrainingArguments(
        num_train_epochs=args.epoch,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=50,
        learning_rate=1e-4,
        weight_decay=0.01,
        logging_steps=5,
        save_strategy="steps",
        save_steps=250,
        save_total_limit=1,
        # optim="adamw_hf", # for 8-bit, keep paged_adamw_8bit, else adamw_hf
        optim="adamw_torch",
        # fp16=True,
        bf16=True,
        output_dir=f"./0713_{model_name}-taisc(strategy3-{args.dataset}-{args.epoch}epoch-complete)",
        hub_model_id=f"{model_name}-taisc",
        remove_unused_columns=False,
        report_to="tensorboard",
        dataloader_pin_memory=False,
    )

    trainer = BinaryClassificationTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=train_smolvlm2,
    )

    trainer.train()

    processor_save_path = get_latest_checkpoint(training_args.output_dir)
    os.makedirs(processor_save_path, exist_ok=True)
    processor.save_pretrained(processor_save_path)
    print(f"Processor 已儲存至：{processor_save_path}")

    # trainer.push_to_hub()

__all__ = ["SmolVLMWithClassifier"]