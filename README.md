# Robust Measurement While Drilling of Surrounding Rock Mass Under Missing Data Conditions by Domain Knowledge Integrated Large Language Models

This repository contains the official implementation of the paper: **"Robust Measurement While Drilling of Surrounding Rock Mass Under Missing Data Conditions by Domain Knowledge Integrated Large Language Models"**.

## 📖 Abstract
Measurement While Drilling (MWD) for surrounding rock quality holds significant importance in engineering practice. However, under complex construction conditions, sensor failures and communication anomalies often lead to missing measurement data, posing challenges to the reliability of MWD. This study aims to propose a robust MWD method for surrounding rock quality under data-incomplete conditions. This method employs large language models integrated with domain knowledge to perform feature extraction and information fusion, overcoming the limitations of traditional oversampling techniques when handling extremely sparse samples. Furthermore, by enabling collaborative decision-making between the large language models and the traditional machine learning method XGBoost, the accuracy of surrounding rock classification is significantly enhanced. Validation on real-world datasets demonstrates that even with a 50% data missing rate, the proposed method achieves 82.60% classification accuracy, outperforming existing data imputation methods. Furthermore, this study validates the critical role of domain knowledge integration in enhancing the reasoning capabilities of large language models.
## 📂 Data Availability

The dataset employed in this study is derived from the open-source research by **Hansen et al. (2024)**, which originates from 15 geologically diverse hard rock tunnels across four infrastructure projects in Norway (UDK, UNB, RV4, and E39).

- **Original Source:** [Predicting rock type from MWD tunnel data (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729646) or related repository.
- **File Name:** Please rename your raw dataset to `mwd_rocktype_blastholes_raw.csv` and place it in the root directory.
- **Knowledge Base:** The project relies on a domain knowledge dictionary. Ensure `english_domain_knowledge.json` is present in the root directory.

*(Note: If you cannot publicly distribute the dataset due to licensing, users can request it from the original authors or use their own MWD data following the same format.)*

## 🤖 LLM Backbone & Setup

This framework utilizes **Qwen-1.8B-Chat** as the foundational Large Language Model due to its balance between inference efficiency and reasoning capability.

### 1. Download the Model
You can download the model weights from Hugging Face or ModelScope (recommended for users in China).

- **Hugging Face:** [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)
- **ModelScope:** [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat)

### 2. Configuration
After downloading, please update the `model_path` variable in `step1_llm_pipeline.py` (line 34) to point to your local model directory:

```python
CONFIG = {
    # ...
    "model_path": "/path/to/your/local/Qwen-1_8B-Chat", 
    # ...
}
