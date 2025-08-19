import os, argparse, datetime, torch, pandas as pd
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
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
    cands = []
    if args.processor_dir:
        cands.append(Path(args.processor_dir))
    try:
        latest = Path(get_latest_checkpoint(str(base)))
        cands.append(latest)
    except Exception:
        pass
    cands += [base / "checkpoint-36", base]
    for c in cands:
        if not c.exists(): continue
        has_tok = (c / "tokenizer.json").exists()
        has_added = (c / "added_tokens.json").exists()
        has_chat = (c / "chat_template.jinja").exists()
        if has_tok and (has_added or has_chat):
            return c
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
if scheme not in ("adapter", "merged"):
    raise ValueError(f"Unexpected scheme value: {scheme}")

# ---------------------- 影片與輸出 -----------------------
video_dir = Path(fr"C:\Python_workspace\TAISC\dataset\{args.dataset}_video\smolvlm2\test")
out_csv   = Path(".") / "program_test" / (
    f"infer_{scheme}_{args.dataset}_{args.epoch}epoch_{datetime.datetime.now():%H-%M-%S}.csv"
)
prompt = "Is a vehicle collision likely to happen within 1 to 2 seconds? (0: No, 1: Yes)"

# ---------------------- 建立模型骨架 ----------------------
model_id = ("HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
            if args.smol else
            "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

if scheme == "adapter":
    # === 方案A：正規做法：用 from_pretrained 建出 PeftModel，然後塞回去 ===
    # 1) 先建你的外層（為了 classifier 與 forward），但會先帶一個暫時的 base
    model = SmolVLMWithClassifier(model_id=model_id, USE_QLORA=True, infer=True).to(device).eval()

    # 2) 用 4-bit 配置重新建「乾淨的 base」
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    clean_base = AutoModelForImageTextToText.from_pretrained(
        model_id, quantization_config=bnb_cfg, device_map="auto"
    )

    # 3) 用 PEFT 官方 API 把 adapter 套到乾淨的 base
    if not adapter_path.exists():
        raise FileNotFoundError(f"找不到 {adapter_path}")
    peft_model = PeftModel.from_pretrained(clean_base, str(adapter_dir), is_trainable=False)

    # 4) 用新的 peft_model 取代外層裡的 base_model（並釋放舊的）
    old = model.base_model
    model.base_model = peft_model
    del old, clean_base
    torch.cuda.empty_cache()

    # 5) 載入分類頭
    if clf_path.exists():
        state_clf = load_file(str(clf_path))
        miss_c, unexp_c = model.load_state_dict(state_clf, strict=False)
        print(f"[classifier] loaded ✓  missing={len(miss_c)}  unexpected={len(unexp_c)}")
    else:
        print("[warn] 找不到 classifier.safetensors，將使用隨機初始化的分類頭（結果可能不準）")

    # 6) 小檢查：應該是 PeftModel、而且具備 adapter
    print("[debug] base type:", type(model.base_model))
    try:
        adapters = list(getattr(model.base_model, "peft_config", {}).keys())
        print("[debug] adapters:", adapters)
    except Exception:
        pass

elif scheme == "merged":
    # === 方案B：合併後整顆 ===
    model = SmolVLMWithClassifier(model_id=model_id, USE_QLORA=False, infer=True).to(device).eval()
    if not merged_path.exists():
        raise FileNotFoundError(f"找不到 {merged_path}")
    state_full = load_file(str(merged_path))
    miss_f, unexp_f = model.load_state_dict(state_full, strict=False)
    print(f"[merged] loaded ✓  missing={len(miss_f)}  unexpected={len(unexp_f)}")

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
            # 避免不必要 cast；若 base 有 dtype 再對齊
            target = getattr(model.base_model, "dtype", None)
            proc_in[k] = v.to(device, dtype=target) if target is not None else v.to(device)
        elif k in ("input_ids", "attention_mask"):
            proc_in[k] = v.to(device, dtype=torch.long)
        else:
            proc_in[k] = v.to(device)

    with torch.inference_mode():
        logit = model(**proc_in).to(torch.float32).squeeze(-1)
        prob  = torch.sigmoid(logit).item()
        pred  = int(prob > 0.5)

    tqdm.write(f"[✓] {name} → p={prob:.5f} ⇒ {pred}")
    results.append({"file_name": name, "risk": pred})

# ---------------------- 存檔 -----------------------------
out_csv = Path(".") / "program_test" / (
    f"infer_{scheme}_{args.dataset}_{args.epoch}epoch_{datetime.datetime.now():%H-%M-%S}.csv"
)
out_csv.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(results).to_csv(out_csv, index=False)
print(f"\n📄 Results saved to: {out_csv}")