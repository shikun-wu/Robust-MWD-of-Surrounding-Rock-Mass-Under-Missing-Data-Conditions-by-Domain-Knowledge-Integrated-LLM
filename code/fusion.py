import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time

EXPERIMENT_ID = 0 
# 0: Full Method 
# 1: No SMOTE 
# 2: No Undersampling 
# 3: XGBoost Only
# 4: Fixed Weights 
# 5: Force Clean Unknowns
# 6: Force Fusion All Classes

DATA_DIR = "dataset_processed"
LABEL_SMOOTHING_FACTOR = 0.1 # Parameter for Post-hoc Probability Estimation

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def make_llm_probs_with_smoothing(encoded_preds, n_classes, smoothing=0.1):
    """
    Converts discrete LLM text predictions into a probability distribution 
    using Post-hoc Label Smoothing. 
    
    Rationale: Since LLM generation yields hard labels, we apply smoothing to 
    estimate confidence, allowing Bayesian Optimization to find finer weights.
    """
    probs = np.zeros((len(encoded_preds), n_classes))
    
    confidence = 1.0 - smoothing
    residual = smoothing / (max(1, n_classes - 1))
    
    for i, pred in enumerate(encoded_preds):
        if pred != -1:
            # Assign smoothed probabilities
            probs[i, :] = residual
            probs[i, pred] = confidence
        else:
            # Maximum Entropy for Unknowns
            probs[i, :] = 1.0 / n_classes
            
    return probs

def main():
    if not os.path.exists(os.path.join(DATA_DIR, "train_final.csv")):
        raise FileNotFoundError("Please run 'step1_llm_process.py' first!")

    print(f"[Info] Loading datasets from {DATA_DIR}...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_final.csv"))
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val_with_llm.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_with_llm.csv"))

    feature_cols = train_df.columns[5:53].tolist() 
    label_col = train_df.columns[4]

    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    

    # 1. Encode Labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[label_col])
    y_val = le.transform(val_df[label_col])
    y_test = le.transform(test_df[label_col])
    class_names = le.classes_
    n_classes = len(class_names)
    
    print(f"[Info] Classes: {n_classes}")

    # 2. Encode LLM Predictions
    known_labels = set(class_names)
    most_common_label = train_df[label_col].mode()[0] 

    def encode_llm_col(series):
        encoded = []
        for x in series:
            x = str(x).strip()
            if x in known_labels:
                encoded.append(le.transform([x])[0])
            else:
                if EXPERIMENT_ID == 5:
                    encoded.append(le.transform([most_common_label])[0])
                else:
                    encoded.append(-1) 
        return np.array(encoded)

    val_llm_encoded = encode_llm_col(val_df['llm_prediction'])
    test_llm_encoded = encode_llm_col(test_df['llm_prediction'])

    # 3. Generate Smoothed Probabilities for LLM
    print("[Info] Applying Post-hoc Label Smoothing to LLM outputs...")
    llm_probs_val = make_llm_probs_with_smoothing(val_llm_encoded, n_classes, LABEL_SMOOTHING_FACTOR)
    llm_probs_test = make_llm_probs_with_smoothing(test_llm_encoded, n_classes, LABEL_SMOOTHING_FACTOR)

    # 4. Undersampling + SMOTE
    print("[Process] Balancing training data for XGBoost...")
    X_train = train_df[feature_cols]
    
    # Undersampling Majority
    class_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
    target_count = class_counts.iloc[1] if len(class_counts) > 1 else class_counts.iloc[0]

    if EXPERIMENT_ID != 2: 
        sampled_indices = []
        for cls_idx in range(n_classes):
            indices = np.where(y_train == cls_idx)[0]
            if len(indices) > target_count:
                selected = np.random.choice(indices, target_count, replace=False)
                sampled_indices.extend(selected)
            else:
                sampled_indices.extend(indices)
        X_train_under = X_train.iloc[sampled_indices]
        y_train_under = y_train[sampled_indices]
    else:
        X_train_under = X_train
        y_train_under = y_train

    # SMOTE Oversampling
    if EXPERIMENT_ID != 1:
        min_samples = pd.Series(y_train_under).value_counts().min()
        k = min(5, min_samples - 1)
        if k > 0:
            smote = SMOTE(random_state=42, k_neighbors=k)
            X_train_bal, y_train_bal = smote.fit_resample(X_train_under, y_train_under)
        else:
            X_train_bal, y_train_bal = X_train_under, y_train_under
    else:
        X_train_bal, y_train_bal = X_train_under, y_train_under

    # 5.XGBoost
    print("[Training] XGBoost Classifier...")
    start_time = time.time()
    
    model_xgb = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6, 
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='multi:softprob', 
        random_state=42,
        n_jobs=-1
    )

    model_xgb.fit(
        X_train_bal, y_train_bal,
        eval_set=[(val_df[feature_cols], y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    training_time = time.time() - start_time
    
    xgb_probs_val = model_xgb.predict_proba(val_df[feature_cols])
    xgb_probs_test = model_xgb.predict_proba(test_df[feature_cols])

    # 6. Fusion
    print("[Strategy] Determining Fusion Mask based on Validation Set...")
    val_pred_xgb = np.argmax(xgb_probs_val, axis=1)
    val_pred_llm = np.argmax(llm_probs_val, axis=1)
    
    fusion_mask = np.zeros(n_classes, dtype=bool)
    
    print(f"{'Class Name':<25} | {'XGB Acc':<10} | {'LLM Acc':<10} | {'Action'}")
    print("-" * 65)

    for i, cls_name in enumerate(class_names):
        indices = np.where(y_val == i)[0]
        if len(indices) == 0: continue
            
        acc_xgb = accuracy_score(y_val[indices], val_pred_xgb[indices])
        valid_llm_indices = [idx for idx in indices if val_llm_encoded[idx] != -1]
        if valid_llm_indices:
             acc_llm = accuracy_score(y_val[valid_llm_indices], val_llm_encoded[valid_llm_indices])
        else:
             acc_llm = 0.0
        
        
        if (acc_llm > acc_xgb and EXPERIMENT_ID != 3) or EXPERIMENT_ID == 6:
            fusion_mask[i] = True
            strategy = "Fusion"
        else:
            fusion_mask[i] = False
            strategy = "XGBoost"
            
        print(f"{cls_name:<25} | {acc_xgb:.4f}     | {acc_llm:.4f}     | {strategy}")

    # 7. Bayesian Optimization
    best_weight_llm = 0.0
    
    if EXPERIMENT_ID == 3 or not np.any(fusion_mask):
        print("\n[Info] Using XGBoost only.")
    elif EXPERIMENT_ID == 4:
        best_weight_llm = 0.5
        print("\n[Info] Using Fixed Weight: 0.5")
    else:
        print("\n[Optimization] Running Bayesian Optimization for Fusion Weights...")
        
        def fusion_score(w_llm):
            w_xgb = 1.0 - w_llm
            final_p = np.zeros_like(xgb_probs_val)
            for c in range(n_classes):
                if fusion_mask[c]:
                    # Probabilistic Fusion 
                    final_p[:, c] = w_llm * llm_probs_val[:, c] + w_xgb * xgb_probs_val[:, c]
                else:
                    final_p[:, c] = xgb_probs_val[:, c]
            
            preds = np.argmax(final_p, axis=1)
            return f1_score(y_val, preds, average='macro')

        optimizer = BayesianOptimization(
            f=fusion_score,
            pbounds={'w_llm': (0.1, 0.9)},
            random_state=42,
            verbose=0,
            allow_duplicate_points=True
        )
        optimizer.maximize(init_points=5, n_iter=20)
        best_weight_llm = optimizer.max['params']['w_llm']
        print(f"[Result] Best LLM Weight: {best_weight_llm:.4f}")

    # 8. Final Prediction
    print("\n[Prediction] Generating Final Test Predictions...")
    final_test_probs = np.zeros_like(xgb_probs_test)
    w_xgb = 1.0 - best_weight_llm

    for c in range(n_classes):
        if fusion_mask[c] and EXPERIMENT_ID != 3:
            final_test_probs[:, c] = best_weight_llm * llm_probs_test[:, c] + w_xgb * xgb_probs_test[:, c]
        else:
            final_test_probs[:, c] = xgb_probs_test[:, c]

    row_sums = final_test_probs.sum(axis=1)
    row_sums[row_sums == 0] = 1.0
    final_test_probs = final_test_probs / row_sums[:, np.newaxis]
    
    final_pred = np.argmax(final_test_probs, axis=1)

    # 9. Evaluation
    print(f"\n===== Experiment {EXPERIMENT_ID} Results =====")
    acc = accuracy_score(y_test, final_pred)
    macro_f1 = f1_score(y_test, final_pred, average='macro')
    
    print(f"Training Time : {training_time:.2f} s")
    print(f"Accuracy      : {acc:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, final_pred, target_names=class_names, digits=4))
    
    plot_confusion_matrix(
        y_test, final_pred, class_names, 
        title="Final Confusion Matrix", 
        save_path="confusion_matrix.png"
    )

if __name__ == "__main__":
    main()
