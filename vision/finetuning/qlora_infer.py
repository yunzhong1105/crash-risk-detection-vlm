# """
# 推論流程：
# 1. 透過 get_latest_checkpoint() 找到包含
#      - model.safetensors        (4-bit 基底)
#      - adapter_model.safetensors(LoRA + classifier)
#    的資料夾
# 2. 建 SmolVLMWithClassifier(USE_QLORA=True)  ⇒ 已自動把 4-bit 基底載好
# 3. 只把 adapter_model.safetensors 權重覆蓋進去
# 4. 影片讀取 & Processor 與舊版 custom_infer.py 完全一致
# """

# import os, argparse, datetime, torch, pandas as pd
# from tqdm import tqdm
# from safetensors.torch import load_file
# from transformers import AutoProcessor
# from smolvlm2_video_FT_lora import SmolVLMWithClassifier, get_latest_checkpoint

# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # ──────────────── CLI 參數 ─────────────────────────────
# def parse_args():
#     parser = argparse.ArgumentParser("Dataset inference script (QLoRA)")
#     parser.add_argument("--dataset", required=True, help="freeway / road / …")
#     parser.add_argument("--epoch", default=3, help="tag used in folder name")
#     parser.add_argument("--model-path", required=True, help="root where checkpoints live")
#     # parser.add_argument("--qlora", action="store_true", help="set True if the checkpoint使用QLoRA(預設 False)")
#     parser.add_argument("--smol", action="store_true", help="True → 500 M, False → 2.2 B")
#     parser.add_argument("--scheme", choices=["auto", "adapter", "merged"], default="auto",
#                     help="adapter=方案A, merged=方案B, auto=自動偵測")
#     parser.add_argument("--processor-dir", default="", help="可選；指定 tokenizer/processor 目錄")

#     return parser.parse_args()

# args = parse_args()

# ckpt_dir = get_latest_checkpoint(args.model_path)
# video_dir = fr"C:\Python_workspace\TAISC\dataset\{args.dataset}_video\smolvlm2\test"
# out_csv = ".\\program_test\\qlora_{}_{}epoch_infer_results_{}.csv".format(args.dataset , args.epoch , str(datetime.datetime.now())[11:19].replace(":" , "-"))
# # out_csv = (f".\\program_test\\full_{args.dataset}_{args.epoch}epoch_"
# #                f"infer_{datetime.datetime.now():%H-%M-%S}.csv")
# prompt = "Is a vehicle collision likely to happen within 1 to 2 seconds?(0: No, 1: Yes)"

# # ──────────────── 建立模型骨架 ─────────────────────────
# model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if args.smol else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

# model = SmolVLMWithClassifier(
#     model_id=model_id,
#     USE_QLORA=args.qlora,     # True → 建 LoRA adapter 結構
#     infer=True
# ).to(device).eval()

# # ──────────────── 只載 LoRA + classifier 權重 ─────────
# if args.qlora:
#     adapter_path = os.path.join(ckpt_dir, "adapter_model.safetensors")
#     state = load_file(adapter_path)
#     miss, unexp = model.load_state_dict(state, strict=False)
#     print(f"[LoRA] loaded  ✓  missing={len(miss)}  unexpected={len(unexp)}")
# else:                           # 全量微調時才讀 model.safetensors
#     full_path = os.path.join(ckpt_dir, "model.safetensors")
#     state = load_file(full_path)
#     model.load_state_dict(state)
#     print("[full] model.safetensors loaded")

# # ──────────────── Processor ───────────────────────────
# processor = AutoProcessor.from_pretrained(ckpt_dir)

# # ──────────────── 影片清單 ────────────────────────────
# vid_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
# vid_paths = [os.path.join(video_dir, f) for f in vid_files]

# results = []
# for path, name in tqdm(list(zip(vid_paths, vid_files)), desc="infer"):
#     messages = [{"role": "user",
#                  "content": [{"type": "text",  "text": prompt},
#                              {"type": "video", "path": path}]}]

#     inputs = processor.apply_chat_template(
#         messages, tokenize=True, add_generation_prompt=False,
#         return_tensors="pt", return_dict=True
#     )

#     proc_in = {}
#     for k, v in inputs.items():
#         if not isinstance(v, torch.Tensor):  # 不是 tensor 直接跳過
#             continue
#         if k == "pixel_values":
#             proc_in[k] = v.to(device, dtype=model.base_model.dtype)
#         elif k in ["input_ids", "attention_mask"]:
#             proc_in[k] = v.to(device, dtype=torch.long)
#         else:
#             proc_in[k] = v.to(device)

#     with torch.inference_mode():
#         logit = model(**proc_in).squeeze(-1)
#         p     = torch.sigmoid(logit).item()
#         pred  = int(p > 0.5)

#     tqdm.write(f"[✓] {name} → p={p:.3f} ⇒ {pred}")
#     results.append({"file_name": name, "risk": pred})

# # ──────────────── 存結果 ──────────────────────────────
# pd.DataFrame(results).to_csv(out_csv, index=False)
# print(f"\n📄 All results saved to: {out_csv}")


"""
支援兩種推論方案：
A. 分離 (adapter + classifier)   → 以 final/adapter + final/classifier.safetensors 為主
B. 合併 (merged single model)     → 以 final/model.safetensors 為主

同時自動挑選 tokenizer/processor：優先使用包含 added_tokens.json、chat_template.jinja 的資料夾
（通常是 checkpoint-XX/），也可用 --processor-dir 覆寫。
"""

import os, argparse, datetime, torch, pandas as pd
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoProcessor
from smolvlm2_video_FT_lora import SmolVLMWithClassifier, get_latest_checkpoint

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser("Dataset inference script (QLoRA, A/B)")
    parser.add_argument("--dataset", required=True, help="freeway / road / …")
    parser.add_argument("--epoch", default=3, help="tag used in filename")
    parser.add_argument("--model-path", required=True, help="root where checkpoints/final live")
    parser.add_argument("--smol", action="store_true", help="True → 500M, False → 2.2B")
    parser.add_argument("--scheme", choices=["auto", "adapter", "merged"], default="auto",
                    help="adapter=方案A, merged=方案B, auto=自動偵測")
    parser.add_argument("--processor-dir", default="", help="可選；指定 tokenizer/processor 目錄")
    return parser.parse_args()

args = parse_args()
root = Path(args.model_path)

# ---------------------- 探測檔案結構 ----------------------
def find_final_dir(base: Path) -> Path | None:
    cand = [base / "final", base]
    for c in cand:
        if (c / "adapter" / "adapter_model.safetensors").exists() or (c / "model.safetensors").exists():
            return c
    return None

def find_checkpoint_dir(base: Path) -> Path | None:
    # 優先：使用帶有 tokenizer 與 chat_template 的資料夾
    cands = []
    if args.processor_dir:
        cands.append(Path(args.processor_dir))
    # 最新 checkpoint
    try:
        latest = Path(get_latest_checkpoint(str(base)))
        cands.append(latest)
    except Exception:
        pass
    # 常見位置
    cands += [base / "checkpoint-36", base]
    # 過濾出含 tokenizer.json 的目錄
    for c in cands:
        if not c.exists():
            continue
        has_tok = (c / "tokenizer.json").exists()
        has_added = (c / "added_tokens.json").exists()
        has_chat = (c / "chat_template.jinja").exists()
        if has_tok and (has_added or has_chat):
            return c
    # 後備：任何含 tokenizer.json 的目錄
    for c in cands:
        if (c / "tokenizer.json").exists():
            return c
    return None

final_dir = find_final_dir(root)
proc_dir  = find_checkpoint_dir(root)
if final_dir is None:
    raise FileNotFoundError("找不到 final/ 或 model 檔案所在的目錄，請確認 --model-path。")
if proc_dir is None:
    raise FileNotFoundError("找不到 tokenizer/processor 的目錄，請加上 --processor-dir。")

adapter_dir   = final_dir / "adapter"
adapter_path  = adapter_dir / "adapter_model.safetensors"
clf_path      = final_dir / "classifier.safetensors"
merged_path   = final_dir / "model.safetensors"

# ---------------------- 決定方案 -------------------------
scheme = args.scheme
if scheme == "auto":
    if adapter_path.exists() and clf_path.exists():
        scheme = "adapter"
    elif merged_path.exists():
        scheme = "merged"
    else:
        raise FileNotFoundError("auto 模式找不到可用的檔案：缺 adapter+classifier 或 merged model。")
print(f"[info] 使用方案：{scheme.upper()}  |  final_dir={final_dir}  |  processor_dir={proc_dir}")

# ---------------------- 影片與輸出 -----------------------
video_dir = Path(fr"C:\Python_workspace\TAISC\dataset\{args.dataset}_video\smolvlm2\test")
out_csv   = Path(".") / "program_test" / (
    f"infer_{scheme}_{args.dataset}_{args.epoch}epoch_{datetime.datetime.now():%H-%M-%S}.csv"
)

prompt = "Is a vehicle collision likely to happen within 1 to 2 seconds? (0: No, 1: Yes)"

# ---------------------- 建立模型骨架 ----------------------
model_id = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct" if args.smol else "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

if scheme == "adapter":
    # A 方案：需要 LoRA 結構（4-bit 量化），待會載入 adapter ΔW + classifier
    model = SmolVLMWithClassifier(
        model_id=model_id,
        USE_QLORA=True,
        infer=True
    ).to(device).eval()
    # 載入 LoRA ΔW → 直接丟到 base_model（PeftModel）
    if adapter_path.exists():
        state_adapter = load_file(str(adapter_path))
        miss_a, unexp_a = model.base_model.load_state_dict(state_adapter, strict=False)
        print(f"[adapter] loaded ✓  missing={len(miss_a)}  unexpected={len(unexp_a)}")
    else:
        raise FileNotFoundError(f"找不到 {adapter_path}")
    # 載入分類頭
    if clf_path.exists():
        state_clf = load_file(str(clf_path))
        miss_c, unexp_c = model.load_state_dict(state_clf, strict=False)
        print(f"[classifier] loaded ✓  missing={len(miss_c)}  unexpected={len(unexp_c)}")
    else:
        print("[warn] 找不到 classifier.safetensors，將使用隨機初始化的分類頭（結果可能不準）")

elif scheme == "merged": 
    # B 方案：不需要 LoRA 結構，直接讀合併後全權重
    model = SmolVLMWithClassifier(
        model_id=model_id,
        USE_QLORA=False,    # 不插 LoRA
        infer=True
    ).to(device).eval()
    if merged_path.exists():
        state_full = load_file(str(merged_path))
        miss_f, unexp_f = model.load_state_dict(state_full, strict=False)
        print(f"[merged] loaded ✓  missing={len(miss_f)}  unexpected={len(unexp_f)}")
    else:
        raise FileNotFoundError(f"找不到 {merged_path}")

else:
    raise RuntimeError(f"Invalid scheme after normalization: {scheme}")


# ---------------------- Processor ------------------------
processor = AutoProcessor.from_pretrained(str(proc_dir))

# ---------------------- 影片清單 -------------------------
vid_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
vid_paths = [video_dir / f for f in vid_files]
if not vid_paths:
    raise FileNotFoundError(f"在 {video_dir} 找不到 mp4 影片。")

# ---------------------- 推論迴圈 ------------------------
results = []
for path, name in tqdm(list(zip(vid_paths, vid_files)), desc="infer"):
    messages = [{"role": "user",
                 "content": [{"type": "text",  "text": prompt},
                             {"type": "video", "path": str(path)}]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_tensors="pt", return_dict=True
    )

    proc_in = {}
    for k, v in inputs.items():
        if not isinstance(v, torch.Tensor):
            continue
        if k == "pixel_values":
            # 若 base_model 有 dtype 就對齊，否則沿用原 dtype（避免不必要 cast）
            target = getattr(model.base_model, "dtype", None)
            proc_in[k] = v.to(device, dtype=target) if target is not None else v.to(device)
        elif k in ("input_ids", "attention_mask"):
            proc_in[k] = v.to(device, dtype=torch.long)
        else:
            proc_in[k] = v.to(device)

    with torch.inference_mode():
        logit = model(**proc_in).squeeze(-1)
        prob  = torch.sigmoid(logit).item()
        pred  = int(prob > 0.5)

    tqdm.write(f"[✓] {name} → p={prob:.5f} ⇒ {pred}")
    # file_name = str(path).split("\\")[-1]
    # tqdm.write(f"[✓] {file_name} → risk: {pred}, logits: {logit[0]:<9.6f}, probs: {prob[0]:<10.8f}")
    results.append({"file_name": name, "risk": pred})

# ---------------------- 存檔 -----------------------------
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n📄 Results saved to: {out_csv}")
