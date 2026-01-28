# Robust Measurement While Drilling of Surrounding Rock Mass Under Missing Data Conditions by Domain Knowledge Integrated Large Language Models

This repository contains the official implementation of the paper: **"Robust Measurement While Drilling of Surrounding Rock Mass Under Missing Data Conditions by Domain Knowledge Integrated Large Language Models"**.

## 📖 Abstract
Measurement While Drilling (MWD) for surrounding rock quality holds significant importance in engineering practice. However, under complex construction conditions, sensor failures and communication anomalies often lead to missing measurement data, posing challenges to the reliability of MWD. This study aims to propose a robust MWD method for surrounding rock quality under data-incomplete conditions. This method employs large language models integrated with domain knowledge to perform feature extraction and information fusion, overcoming the limitations of traditional oversampling techniques when handling extremely sparse samples. Furthermore, by enabling collaborative decision-making between the large language models and the traditional machine learning method XGBoost, the accuracy of surrounding rock classification is significantly enhanced. Validation on real-world datasets demonstrates that even with a 50% data missing rate, the proposed method achieves 82.60% classification accuracy, outperforming existing data imputation methods. Furthermore, this study validates the critical role of domain knowledge integration in enhancing the reasoning capabilities of large language models.
## 📂 Data Availability

The dataset employed in this study is derived from the open-source research by **Hansen et al. (2024)**

- **Original Source:** [Predicting rock type from MWD tunnel data (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4729646) or related repository.


## 🤖 LLM Backbone & Setup

This framework utilizes **Qwen-1.8B-Chat** as the foundational Large Language Model due to its balance between inference efficiency and reasoning capability.

You can download the model weights from Hugging Face or ModelScope (recommended for users in China).

- **Hugging Face:** [Qwen/Qwen-1_8B-Chat](https://huggingface.co/Qwen/Qwen-1_8B-Chat)
- **ModelScope:** [qwen/Qwen-1_8B-Chat](https://modelscope.cn/models/qwen/Qwen-1_8B-Chat)
