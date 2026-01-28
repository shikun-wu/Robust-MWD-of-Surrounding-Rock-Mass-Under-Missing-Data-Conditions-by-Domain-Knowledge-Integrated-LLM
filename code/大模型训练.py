import torch
import numpy as np
import pandas as pd
import json
import os
import shutil
import re
from scipy.ndimage import gaussian_filter1d  
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, PeftModel


torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

KNOWN_CLASSES = [
    "Amphibolittic_gneiss", "Augen_gneiss", "Basalt", "Blackshale", "Breccia",
    "Diabase", "Drammensgranite", "Elnes_lime_shale", "Galgeberg_Hagaberg", "Granite",
    "Granittic_gneiss", "Gravel", "Hagaberg_shale", "Hornfels", "Huk_Hagaberg", "Huk_limestone",
    "Mænaitt", "Pegmatite", "Rhomb_porphyry", "Syenite"
]

CONFIG = {
    "seed": 42,
    "raw_data": "mwd_rocktype_blastholes_raw.csv",
    "output_dir": "dataset_processed",
    "model_path": "qwen/Qwen-1_8B-Chat", # Make sure this path is correct for your local setup
    "lora_path": "rock_lora_weights",
    "finetune_json": "finetune_data.json",
    "missing_ratio": 0.2,
    "knn_k": 4,
    "gaussian_sigma": 1.0  # Parameter for Gaussian Filter
}

KNOWLEDGE_FILE = "english_domain_knowledge.json"

# Load Domain Knowledge
if os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, 'r', encoding='utf-8') as f:
        KNOWLEDGE_DICTIONARY = json.load(f)
    print(f"[Info] Loaded external knowledge from: {KNOWLEDGE_FILE}")
else:
    print(f"[Warning] {KNOWLEDGE_FILE} not found. Using default descriptions.")
    KNOWLEDGE_DICTIONARY = {}

def get_knowledge(rock_type):
    return KNOWLEDGE_DICTIONARY.get(str(rock_type), "Typical hard rock formation found in tunnel excavation.")

# Feature Groups for Block Missing Simulation
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

np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])

# Data Processing
def load_preprocess_split(data_path):
    print(f"[Process] Loading data from: {data_path}")
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    elif data_path.endswith('.xlsx'):
        df = pd.read_excel(data_path)
    else:
        raise ValueError("Only csv and xlsx are supported.")

    if df.shape[1] < 53:
        raise ValueError(f"Insufficient columns. Expected at least 53, got {df.shape[1]}")

    label_col = df.columns[4]
    feature_cols = df.columns[5:53].tolist()
    

    print(f"[Process] Applying Gaussian Filter (sigma={CONFIG['gaussian_sigma']}) to features...")
    for col in feature_cols:
        df[col] = gaussian_filter1d(df[col], sigma=CONFIG['gaussian_sigma'], mode='nearest')

    # Split Data 6:2:2
    train_df, temp_df = train_test_split(
        df, test_size=0.4, stratify=df[label_col], random_state=CONFIG["seed"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df[label_col], random_state=CONFIG["seed"]
    )

    print(f"[Info] Split complete: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df, feature_cols, label_col

def inject_missing_values(df, feature_groups, target_ratio=0.2):
    df_missing = df.copy()
    all_group_features = [f for group in feature_groups for f in group]
    
    n_rows = len(df_missing)
    total_cells = n_rows * len(all_group_features)
    
    print(f"[Process] Injecting missing values... Target Ratio: {target_ratio:.1%}")
    
    count_missing = 0
    for idx in df_missing.index:
        for group in feature_groups:
            if np.random.random() < target_ratio:
                df_missing.loc[idx, group] = np.nan
                count_missing += len(group)
    
    actual_ratio = count_missing / total_cells
    print(f"   -> Actual Missing Ratio: {actual_ratio:.2%}")
    return df_missing

def apply_knn_imputation(train_df, val_df, test_df, feature_cols, n_neighbors=4):
    print(f"[Process] Applying KNN Imputation (k={n_neighbors})...")
    
    imputer = KNNImputer(n_neighbors=n_neighbors)
    
    X_train = train_df[feature_cols].values
    X_val = val_df[feature_cols].values
    X_test = test_df[feature_cols].values
    
    imputer.fit(X_train)
    
    train_filled = train_df.copy()
    val_filled = val_df.copy()
    test_filled = test_df.copy()
    
    train_filled[feature_cols] = imputer.transform(X_train)
    val_filled[feature_cols] = imputer.transform(X_val)
    test_filled[feature_cols] = imputer.transform(X_test)
    
    print("[Info] KNN Imputation complete.")
    return train_filled, val_filled, test_filled


# LLM Finetuning & Generation
def generate_finetune_data(df, feature_cols, label_col, output_file):
    finetune_samples = []
    print(f"[Process] Generating Instruction Tuning Data -> {output_file}")
    
    for _, row in df.iterrows():
        rock_type = row[label_col]
        knowledge_desc = get_knowledge(rock_type)
        
        all_features = [f"{col}: {row[col]:.4f}" for col in feature_cols]
        
        sample = {
            "instruction": f"Predict the rock type based on the provided MWD features. Reference Knowledge: {knowledge_desc}. Return only the class name.",
            "input": f"Features: {'; '.join(all_features)}",
            "output": f"{row[label_col]}"
        }
        finetune_samples.append(sample)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(finetune_samples, f, ensure_ascii=False, indent=2)
    
    print(f"[Info] Generated {len(finetune_samples)} samples.")

def finetune_llm(base_model_path, finetune_data_path, lora_weights_dir):
    print(f"\n[LLM] Starting LoRA Fine-tuning using: {base_model_path}")
    
    if os.path.exists(lora_weights_dir):
        print("[Warning] Cleaning up old LoRA weights...")
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
            f"### Instruction: {inst}\n### Input: {inp}\n### Output: {out}"
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
    print(f"[Success] LoRA weights saved to: {lora_weights_dir}")
    
    del model, trainer
    torch.cuda.empty_cache()

def synthesize_minority_samples_llm_only(df_filled, generator, feature_cols, label_col, min_samples=5):
    print("\n[Synthesis] Starting LLM-based sample generation for rare classes...")
    
    df_temp = df_filled.copy()
    class_counts = df_temp[label_col].value_counts()
    
    synthetic_samples = []
    # Identify classes that strictly need LLM generation (N < 5)
    llm_classes = [cls for cls, cnt in class_counts.items() if cnt < min_samples]

    if not llm_classes:
        print("[Info] No classes below threshold. Skipping LLM synthesis.")
        return df_filled

    for class_label in llm_classes:
        count = class_counts[class_label]
        num_needed = min_samples - count
        print(f"   -> Generating {num_needed} samples for class '{class_label}'...")

        class_data = df_temp[df_temp[label_col] == class_label][feature_cols]
        stats = class_data.describe()
        total_features = len(feature_cols)
        class_knowledge = get_knowledge(class_label)

        for i in range(num_needed):
            # Prompt Engineering: Enforcing strict numerical output format
            prompt = f"""### Instruction:
Generate a synthetic MWD data sample for rock type '{class_label}'.
Requirements:
1. Generate exactly {total_features} numerical values, separated by commas.
2. Values should be within reasonable range (4 decimal places).
3. Output ONLY the comma-separated numbers.

### Domain Knowledge: {class_knowledge}
### Statistical Reference:
{stats.to_string()}
### Output:"""

            try:
                response = generator(prompt, max_new_tokens=300, temperature=0.1, return_full_text=False)
                output_text = response[0]['generated_text'].strip()
                
                output_clean = re.sub(r'[^\d.,-]', '', output_text)
                values = [v.strip() for v in output_clean.split(',') if v.strip()]
                
                valid_values = []
                for v in values:
                    try: valid_values.append(round(float(v), 4))
                    except: continue

                # Padding or Truncating to match feature dimensions
                if len(valid_values) > total_features:
                    valid_values = valid_values[:total_features]
                elif len(valid_values) < total_features:
                    missing = total_features - len(valid_values)
                    fill_values = [class_data[col].mean() for col in feature_cols[-missing:]]
                    valid_values += fill_values
                    valid_values = [0.0 if np.isnan(v) else v for v in valid_values]

                if len(valid_values) == total_features:
                    synthetic = dict(zip(feature_cols, valid_values))
                    synthetic[label_col] = class_label
                    synthetic_samples.append(synthetic)
                else:
                    print(f"      [Error] Dimension mismatch ({len(valid_values)}), skipping.")
            except Exception as e:
                print(f"      [Error] Generation failed: {e}")

    if synthetic_samples:
        synthetic_df = pd.DataFrame(synthetic_samples)
        df_augmented = pd.concat([df_filled, synthetic_df], ignore_index=True)
        print(f"[Success] Synthesized {len(synthetic_samples)} new samples using LLM.")
        return df_augmented
    
    return df_filled

def batch_predict(model, tokenizer, df, feature_cols, batch_size=32):
    model.eval()
    predictions = []
    dataset = Dataset.from_pandas(df.reset_index(drop=True))
    total_samples = len(df)
    
    # Sort classes by length to avoid partial matching issues
    sorted_classes = sorted(KNOWN_CLASSES, key=lambda x: len(x), reverse=True)

    def generate_prompts(batch):
        prompts = []
        for i in range(len(batch[feature_cols[0]])):
            features = [f"{col}: {batch[col][i]:.4f}" for col in feature_cols]
            prompt = f"""### Instruction: Classify the rock type based on the features.
Candidates: {', '.join(KNOWN_CLASSES)}
### Features: {'; '.join(features)}
### Output:"""
            prompts.append(prompt)
        return {"prompt": prompts}

    dataset = dataset.map(generate_prompts, batched=True, batch_size=batch_size)
    print(f"[Inference] Predicting {total_samples} samples...")
    
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
        print(f"[Error] Inference failed: {e}")
        predictions = ["Error"] * total_samples

    return predictions

def main(retrain=False):
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # 1. Load, Gaussian Filter, Split
    train_df, val_df, test_df, feature_cols, label_col = load_preprocess_split(CONFIG["raw_data"])

    # 2. Simulate Missing Values
    train_missing = inject_missing_values(train_df, FEATURE_GROUPS, CONFIG["missing_ratio"])
    val_missing = inject_missing_values(val_df, FEATURE_GROUPS, CONFIG["missing_ratio"])
    test_missing = inject_missing_values(test_df, FEATURE_GROUPS, CONFIG["missing_ratio"])

    # 3. KNN Imputation
    train_filled, val_filled, test_filled = apply_knn_imputation(
        train_missing, val_missing, test_missing, feature_cols, CONFIG["knn_k"]
    )

    # 4. Generate Fine-tuning Data
    json_path = os.path.join(CONFIG["output_dir"], CONFIG["finetune_json"])
    if not os.path.exists(json_path):
        generate_finetune_data(train_filled, feature_cols, label_col, json_path)
    
    # 5. LoRA
    if retrain or not os.path.exists(CONFIG["lora_path"]):
        finetune_llm(CONFIG["model_path"], json_path, CONFIG["lora_path"])
    else:
        print(f"[Info] Found existing LoRA weights at {CONFIG['lora_path']}, skipping training.")

    # 6. Synthesis and Inference
    print("\n[LLM] Loading model for Inference/Synthesis...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_path"], trust_remote_code=True, torch_dtype=torch.float16, device_map="cuda"
    )
    model = PeftModel.from_pretrained(base_model, CONFIG["lora_path"])
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    
    # 7. Synthesize Samples 
    train_augmented = synthesize_minority_samples_llm_only(
        train_filled, generator, feature_cols, label_col, min_samples=5
    )

    # 8. Inference on Val and Test
    print("\n[Inference] Running LLM prediction on Validation Set...")
    val_filled['llm_prediction'] = batch_predict(model, tokenizer, val_filled, feature_cols)
    
    print("\n[Inference] Running LLM prediction on Test Set...")
    test_filled['llm_prediction'] = batch_predict(model, tokenizer, test_filled, feature_cols)

    # 9. Save Results
    train_augmented.to_csv(os.path.join(CONFIG["output_dir"], "train_final.csv"), index=False)
    val_filled.to_csv(os.path.join(CONFIG["output_dir"], "val_with_llm.csv"), index=False)
    test_filled.to_csv(os.path.join(CONFIG["output_dir"], "test_with_llm.csv"), index=False)

    print("\n[Done] Step 1 Complete. Files saved to:", CONFIG["output_dir"])

if __name__ == "__main__":
    main(retrain=False)
