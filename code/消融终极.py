import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from bayes_opt import BayesianOptimization
from imblearn.over_sampling import SMOTE
import time
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 混淆矩阵
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


# 0: 完整流程 
# 1: 移除SMOTE 
# 2: 移除欠采样 
# 3: 纯XGBoost
# 4: 固定权重融合 
# 5: 强制清洗LLM未知预测
# 6: 强制融合所有类别 
experiment_id = 0 

# 1. 数据加载
DATA_DIR = "dataset_split"
if not os.path.exists(os.path.join(DATA_DIR, "train_final.csv")):
    raise FileNotFoundError("请先运行生成划分好的数据文件！")

print(f"正在加载数据集: {DATA_DIR} ...")
train_df = pd.read_csv(os.path.join(DATA_DIR, "train_final.csv"))
val_df = pd.read_csv(os.path.join(DATA_DIR, "val_with_llm.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test_with_llm.csv"))

feature_cols = train_df.columns[5:53].tolist()
label_col = train_df.columns[4]

print(f"训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")

# 2. 高斯滤波
print("正在应用高斯滤波降噪...")
for df in [train_df, val_df, test_df]:
    for col in feature_cols:
        df[col] = gaussian_filter1d(df[col], sigma=1.0, mode='nearest')

# 3. 标签与LLM预测编码
le = LabelEncoder()
y_train = le.fit_transform(train_df[label_col])
y_val = le.transform(val_df[label_col])
y_test = le.transform(test_df[label_col])
class_names = le.classes_
n_classes = len(class_names)

# 如果实验5，则清洗未知类别；否则标记为-1
known_labels = set(class_names)
most_common_label = train_df[label_col].mode()[0] 

def encode_llm_col(series):
    encoded = []
    for x in series:
        x = str(x).strip()
        if x in known_labels:
            encoded.append(le.transform([x])[0])
        else:
            if experiment_id == 5:
                encoded.append(le.transform([most_common_label])[0])
            else:
                encoded.append(-1) 
    return np.array(encoded)

val_llm_encoded = encode_llm_col(val_df['llm_prediction'])
test_llm_encoded = encode_llm_col(test_df['llm_prediction'])

# 生成 LLM 的 One-Hot 概率矩阵 
def make_llm_probs(encoded_preds, n_classes):
    probs = np.zeros((len(encoded_preds), n_classes))
    for i, pred in enumerate(encoded_preds):
        if pred != -1:
            probs[i, pred] = 1.0
        else:
            probs[i, :] = 0.0 
    return probs

llm_probs_val = make_llm_probs(val_llm_encoded, n_classes)
llm_probs_test = make_llm_probs(test_llm_encoded, n_classes)

# 4. 平衡
print("正在处理训练集不平衡...")
X_train = train_df[feature_cols]

# 先对多数类欠采样，再对少数类SMOTE
class_counts = pd.Series(y_train).value_counts().sort_values(ascending=False)
target_count = class_counts.iloc[1] if len(class_counts) > 1 else class_counts.iloc[0] # 第二多的数量

# 欠采样 
if experiment_id != 2:
    sampled_indices = []
    for cls in class_names:
        cls_idx = le.transform([cls])[0]
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

# SMOTE过采样 
if experiment_id != 1:
    # 动态计算 k_neighbors
    min_samples = pd.Series(y_train_under).value_counts().min()
    k = min(5, min_samples - 1)
    if k > 0:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_under, y_train_under)
    else:
        X_train_bal, y_train_bal = X_train_under, y_train_under
else:
    X_train_bal, y_train_bal = X_train_under, y_train_under

print(f"处理后训练集形状: {X_train_bal.shape}")

# 5. XGBoost
print("正在训练 XGBoost...")
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


# 6. 制定融合策略 
print("正在验证集上制定融合策略...")

# 计算 XGBoost 和 LLM 在验证集各类别上的 Accuracy
val_pred_xgb = np.argmax(xgb_probs_val, axis=1)
val_pred_llm = np.argmax(llm_probs_val, axis=1) 

fusion_mask = np.zeros(n_classes, dtype=bool) 

print(f"{'类别名称':<25} | {'XGB Acc':<10} | {'LLM Acc':<10} | {'策略'}")
print("-" * 65)

for i, cls_name in enumerate(class_names):
    indices = np.where(y_val == i)[0]
    
    if len(indices) == 0:
        continue
        
    acc_xgb = accuracy_score(y_val[indices], val_pred_xgb[indices])
    
    # 对于LLM，需要排除未知预测(-1)的影响
    acc_llm = accuracy_score(y_val[indices], val_pred_llm[indices])
    
    # 融合
    if (acc_llm > acc_xgb and experiment_id != 3) or experiment_id == 6:
        fusion_mask[i] = True
        strategy = "✅ 融合"
    else:
        fusion_mask[i] = False
        strategy = "   XGB"
        
    print(f"{cls_name:<25} | {acc_xgb:.4f}     | {acc_llm:.4f}     | {strategy}")

# 7. 优化融合权重 (基于验证集，仅针对筛选出的类别)
best_weight_llm = 0.0

if experiment_id == 3:
    print("\n实验3：跳过融合，仅使用 XGBoost")
elif not np.any(fusion_mask):
    print("\n没有类别需要融合，全部使用 XGBoost")
else:
    
    if experiment_id == 4:
        best_weight_llm = 0.5
        print(f"\n实验4：使用固定权重 LLM={best_weight_llm}")
    else:
        print("\n正在使用贝叶斯优化寻找最佳融合权重 (基于验证集)...")
        
        def fusion_score(w_llm):

            
            w_xgb = 1.0 - w_llm
            
            final_probs = xgb_probs_val.copy() * w_xgb 
            

            

            final_p = np.zeros_like(xgb_probs_val)
            for c in range(n_classes):
                if fusion_mask[c]:
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
        print(f"最优 LLM 权重: {best_weight_llm:.4f}")


print("\n正在生成测试集最终预测...")

final_test_probs = np.zeros_like(xgb_probs_test)
w_xgb = 1.0 - best_weight_llm

for c in range(n_classes):
    if fusion_mask[c] and experiment_id != 3:
        final_test_probs[:, c] = best_weight_llm * llm_probs_test[:, c] + w_xgb * xgb_probs_test[:, c]
    else:
        final_test_probs[:, c] = xgb_probs_test[:, c]

row_sums = final_test_probs.sum(axis=1)
row_sums[row_sums == 0] = 1.0
final_test_probs = final_test_probs / row_sums[:, np.newaxis]

final_pred = np.argmax(final_test_probs, axis=1)


print(f"\n===== 实验 {experiment_id} 最终测试集评估 =====")

acc = accuracy_score(y_test, final_pred)
macro_f1 = f1_score(y_test, final_pred, average='macro')
weighted_f1 = f1_score(y_test, final_pred, average='weighted')

print(f"Training Time : {training_time:.2f} s")
print(f"Accuracy      : {acc:.4f}")
print(f"Macro F1      : {macro_f1:.4f}")
print(f"Weighted F1   : {weighted_f1:.4f}")

print("\n详细分类报告:")
print(classification_report(y_test, final_pred, target_names=class_names, digits=4))

# 绘制混淆矩阵
plot_confusion_matrix(
    y_test, final_pred, class_names, 
    title=f"Exp {experiment_id} Confusion Matrix", 
    save_path=f"Exp{experiment_id}_CM.png"
)