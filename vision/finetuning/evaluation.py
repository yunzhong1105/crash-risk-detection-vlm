import json
import pandas as pd
import argparse
from sklearn.metrics import f1_score, accuracy_score

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

def evaluate(label_df, pred_df):
    # 將欄位名稱統一為 "id"
    label_df = label_df.rename(columns={"video": "id"})
    pred_df = pred_df.rename(columns={"file_name": "id", "risk": "pred"})

    # 合併兩個 dataframe
    df = pd.merge(label_df, pred_df, on="id", how="inner")

    if df.empty:
        raise ValueError("Label 與 Prediction 沒有對齊的 id（video/file_name）。")

    y_true = df["label"]
    y_pred = df["pred"]

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return {"accuracy": acc, "f1_score": f1}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label-path', type=str, required=True, help='Path to label JSONL file')
    parser.add_argument('--pred-path', type=str, required=True, help='Path to prediction CSV file')
    args = parser.parse_args()

    label_df = load_jsonl(args.label_path)
    pred_df = pd.read_csv(args.pred_path)

    # 驗證欄位
    if "video" not in label_df.columns or "label" not in label_df.columns:
        raise ValueError("Label 檔案必須包含 'video' 和 'label' 欄位。")
    if "file_name" not in pred_df.columns or "risk" not in pred_df.columns:
        raise ValueError("Prediction 檔案必須包含 'file_name' 和 'score' 欄位。")

    results = evaluate(label_df, pred_df)

    print("\n=== Evaluation Metrics ===")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
