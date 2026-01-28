import torch
import numpy as np
import pandas as pd
import json
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel

# 启用PyTorch GPU内存优化
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

known_classes = [
    "Amphibolittic_gneiss", "Augen_gneiss", "Basalt", "Blackshale", "Breccia",
    "Diabase", "Drammensgranite", "Elnes_lime_shale", "Galgeberg_Hagaberg", "Granite",
    "Granittic_gneiss", "Gravel", "Hagaberg_shale", "Hornfels", "Huk_Hagaberg", "Huk_limestone",
    "Mænaitt", "Pegmatite", "Rhomb_porphyry", "Syenite"
]

FEATURE_GROUPS = [
    ["PenetrNormMean", "PenetrNormMedian", "PenetrNormVariance", "PenetrNormStandardDeviation", "PenetrNormSkewness", "PenetrNormKurtosis"],
    ["PenetrRMSMean", "PenetrRMSMedian", "PenetrRMSVariance", "PenetrRMSStandardDeviation", "PenetrRMSSkewness", "PenetrRMSKurtosis"],
    ["RotaPressNormMean", "RotaPressNormMedian", "RotaPressNormVariance", "RotaPressNormStandardDeviation", "RotaPressNormSkewness", "RotaPressNormKurtosis"],
    ["RotaPressRMSMean", "RotaPressRMSMedian", "RotaPressRMSVariance", "RotaPressRMSStandardDeviation", "RotaPressRMSSkewness", "RotaPressRMSKurtosis"],
    ["FeedPressNormMean", "FeedPressNormMedian", "FeedPressNormVariance", "FeedPressNormStandardDeviation", "FeedPressNormSkewness", "FeedPressNormKurtosis"],
    ["HammerPressNormMean", "HammerPressNormMedian", "HammerPressNormVariance", "HammerPressNormStandardDeviation", "HammerPressNormSkewness", "HammerPressNormKurtosis"],
    ["WaterFlowNormMean", "WaterFlowNormMedian", "WaterFlowNormVariance", "WaterFlowNormStandardDeviation", "WaterFlowNormSkewness", "WaterFlowNormKurtosis"],
    ["WaterFlowRMSMean", "WaterFlowRMSMedian", "WaterFlowRMSVariance", "WaterFlowRMSStandardDeviation", "WaterFlowRMSSkewness", "WaterFlowRMSKurtosis"],
]

np.random.seed(42)
torch.manual_seed(42)

# 1. 数据加载与 6:2:2 划分
def load_and_split_data(data_path):
    print(f"⏳ 正在加载数据: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("仅支持csv和xlsx格式")

    if df.shape[1] < 53:
        raise ValueError(f"数据列数不足，实际只有{df.shape[1]}列")

    label_col = df.columns[4]
    feature_cols = df.columns[5:53].tolist()
    
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df[label_col], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=42
    )

    print(f"✅ 数据划分完成: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df, feature_cols, label_col

# 2. 制造缺失值（按特征组）
def inject_missing_values(df, feature_groups, target_ratio=0.2):
    df_missing = df.copy()
    all_group_features = [f for group in feature_groups for f in group]
    
    n_rows = len(df_missing)
    total_cells = n_rows * len(all_group_features)
    
    print(f"   正在制造缺失值... 目标缺失率: {target_ratio:.1%}")
    
    # 逐行处理：按概率决定是否让某个特征组（6个特征）整体缺失
    prob = target_ratio
    count_missing = 0
    
    for idx in df_missing.index:
        for group in feature_groups:
            if np.random.random() < prob:
                df_missing.loc[idx, group] = np.nan
                count_missing += len(group)
    
    actual_ratio = count_missing / total_cells
    print(f"   实际生成缺失率: {actual_ratio:.2%}")
    return df_missing

# 3. KNN 填补
def apply_knn_imputation(train_df, val_df, test_df, feature_cols, n_neighbors=4):
    print(f"⏳ 正在进行 KNN 填补 (k={n_neighbors})...")
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    # 仅使用特征列进行计算
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    # Fit on Train
    imputer.fit(X_train)
    
    # Transform all
    X_train_filled = imputer.transform(X_train)
    X_val_filled = imputer.transform(X_val)
    X_test_filled = imputer.transform(X_test)
    
    # 重组 DataFrame
    train_filled = train_df.copy()
    val_filled = val_df.copy()
    test_filled = test_df.copy()
    
    train_filled[feature_cols] = X_train_filled
    val_filled[feature_cols] = X_val_filled
    test_filled[feature_cols] = X_test_filled
    
    print("✅ KNN 填补完成 ")
    return train_filled, val_filled, test_filled

# 4. 生成微调数据
def generate_finetune_data(df, feature_cols, label_col, output_file):
    finetune_samples = []
    print(f"⏳ 正在生成微调指令数据 ({output_file})...")
    
    for _, row in df.iterrows():
        all_features = [
            f"{col}: {row[col]:.4f}" for col in feature_cols
        ]
        
        sample = {
            "instruction": "根据提供的岩石特征，预测其所属类别。仅返回类别名称，不添加任何额外说明。",
            "input": f"已知岩石特征：{'; '.join(all_features)}",
            "output": f"{row[label_col]}"
        }
        finetune_samples.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(finetune_samples, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 微调数据生成完毕: {len(finetune_samples)} 条样本")

# 5. 微调模型
def finetune_llm(base_model_path, finetune_data_path, lora_weights_dir):
    print(f"\n🚀 开始加载模型进行微调: {base_model_path}")
    
    # 如果存在旧权重，先清理，防止混合
    if os.path.exists(lora_weights_dir):
        print("⚠️ 检测到旧权重目录，正在清理...")
        shutil.rmtree(lora_weights_dir)
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True, padding_side="right")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
    )

    lora_config = LoraConfig(
        r=16, lora_alpha=64, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    
    dataset = load_dataset("json", data_files=finetune_data_path)['train']
    
    def preprocess_function(examples):
        prompts = [
            f"### 指令：{inst}\n### 输入：{inp}\n### 输出：{out}"
            for inst, inp, out in zip(examples["instruction"], examples["input"], examples["output"])
        ]
        return tokenizer(prompts, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir="./temp_checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=10,
        save_strategy="no",
        fp16=True,
        report_to="none",
        optim="paged_adamw_8bit"
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()
    
    os.makedirs(lora_weights_dir, exist_ok=True)
    model.save_pretrained(lora_weights_dir)
    print(f"✅ LoRA权重已保存至: {lora_weights_dir}")
    
    del model, trainer
    torch.cuda.empty_cache()

# 6. 批量推理
def batch_predict(model, tokenizer, df, feature_cols, batch_size=32):
    model.eval()
    predictions = []
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    total_samples = len(df)
    sorted_classes = sorted(known_classes, key=lambda x: len(x), reverse=True)

    def generate_prompts(batch):
        prompts = []
        for i in range(len(batch[feature_cols[0]])):
            features = [f"{col}: {batch[col][i]:.4f}" for col in feature_cols]
            prompt = f"""### 指令：根据以下岩石特征，从给定类别列表中选择最匹配的类别。
仅返回类别名称。
备选类别：{', '.join(known_classes)}

### 特征：{'; '.join(features)}
### 输出："""
            prompts.append(prompt)
        return {"prompt": prompts}

    dataset = dataset.map(generate_prompts, batched=True, batch_size=batch_size)
    print(f"   正在推理 {total_samples} 条数据...")
    
    generator = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=30, do_sample=False, batch_size=batch_size, device=0
    )

    try:
        results = generator(dataset["prompt"], return_full_text=False)
        for res in results:
            output = res[0]['generated_text'].strip()
            matched = None
            for cls in sorted_classes:
                if output.startswith(cls):
                    matched = cls
                    break
            predictions.append(matched if matched else "Unknown")
    except Exception as e:
        print(f"❌ 推理出错: {e}")
        predictions = ["Error"] * total_samples

    return predictions

# 主程序
def main(retrain=False):
    config = {
        "raw_data": "mwd_rocktype_blastholes_raw.csv",
        "output_dir": "dataset_split",
        "model_path": "qwen-1.8b-chat",
        "lora_path": "rock_lora_weights",
        "finetune_json": "finetune_train_knn.json",
        "missing_ratio": 0.2, # 缺失率设定
        "knn_k": 4
    }
    
    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. 加载并划分数据 (6:2:2)
    train_df, val_df, test_df, feature_cols, label_col = load_and_split_data(config["raw_data"])

    # 2. 制造缺失值 
    print("\n🔨 开始制造缺失值 (模拟传感器故障)...")
    train_missing = inject_missing_values(train_df, FEATURE_GROUPS, config["missing_ratio"])
    val_missing = inject_missing_values(val_df, FEATURE_GROUPS, config["missing_ratio"])
    test_missing = inject_missing_values(test_df, FEATURE_GROUPS, config["missing_ratio"])

    # 3. KNN 填补 
    print("\n🔧 开始 KNN 填补...")
    train_filled, val_filled, test_filled = apply_knn_imputation(
        train_missing, val_missing, test_missing, feature_cols, config["knn_k"]
    )

    # 4. 生成微调数据 
    json_path = os.path.join(config["output_dir"], config["finetune_json"])
    if not os.path.exists(json_path):
        generate_finetune_data(train_filled, feature_cols, label_col, json_path)
    
    # 5. 微调模型 (逻辑控制)
    # 如果 retrain=True 或者 权重目录不存在，则执行微调
    if retrain or not os.path.exists(config["lora_path"]):
        print(f"🔄 触发训练 (Retrain={retrain}, Path Exists={os.path.exists(config['lora_path'])})")
        finetune_llm(config["model_path"], json_path, config["lora_path"])
    else:
        print(f"⚡ 检测到已有LoRA权重且 retrain=False，跳过训练，直接加载权重: {config['lora_path']}")

    # 6. 加载模型进行预测
    print("\n⏳ 加载模型准备预测...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config["model_path"], trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, config["lora_path"])
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 7. 对 Val 和 Test 进行预测 (使用填补后的数据)
    print("\n🔮 对 [验证集] (已KNN修复) 进行预测...")
    val_filled['llm_prediction'] = batch_predict(model, tokenizer, val_filled, feature_cols)
    
    print("\n🔮 对 [测试集] (已KNN修复) 进行预测...")
    test_filled['llm_prediction'] = batch_predict(model, tokenizer, test_filled, feature_cols)

    # 8. 保存文件供下游任务使用
    train_filled.to_csv(os.path.join(config["output_dir"], "train_final.csv"), index=False)
    val_filled.to_csv(os.path.join(config["output_dir"], "val_with_llm.csv"), index=False)
    test_filled.to_csv(os.path.join(config["output_dir"], "test_with_llm.csv"), index=False)

    print("\n✅ 处理完成！")

if __name__ == "__main__":
    main(retrain=False)