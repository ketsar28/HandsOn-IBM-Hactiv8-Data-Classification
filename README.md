# 🧠 Personality Insight AI Agent

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![LangChain](https://img.shields.io/badge/Made%20with-LangChain-blue)](https://python.langchain.com/)
[![Replicate](https://img.shields.io/badge/Powered%20by-Replicate-29a3d5?logo=replicate)](https://replicate.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

A data science and AI integration project combining **psychological insights**, **machine learning**, and **LLM-based agents** to explore and interact with personality data using natural language. This notebook runs entirely in Google Colab and leverages tools like **LangChain**, **Replicate**, and **IBM Granite Models**.

---

## 👨‍🏫 Project Background

This project was inspired by an educational initiative that encourages learners to combine traditional data science techniques with modern large language models (LLMs). Using free credits and access to cutting-edge tools like **IBM Granite Foundation Models** through **Replicate**, this notebook explores the intersection between structured data and natural language interaction.

---

## 🔍 Project Overview

This notebook walks through:

- ✅ Data analysis on a synthetic **personality classification dataset**
- 📊 Visualization and feature exploration
- 🧠 Classification models (e.g., SVM, Decision Tree)
- 🤖 AI Agent using **LangChain + LLM** to query the dataset
- 🔗 Integration with **IBM Granite LLM** via Replicate
- 💬 Example: _"What is the most common trait of Ambiverts?"_

---

## 📁 Dataset Used

- **Dataset**: [Introvert, Extrovert and Ambivert Classification (Kaggle)](https://www.kaggle.com/datasets/miadul/introvert-extrovert-and-ambivert-classification)  
- **Type**: Synthetic behavioral dataset  
- **Columns**: Emotional, social, and behavioral metrics  
- **Target**: Personality type (Introvert, Extrovert, Ambivert)  

---

## ⚙️ Tools & Technologies

| Category            | Tools / Platforms |
|---------------------|-------------------|
| IDE & Runtime        | [Google Colab](https://colab.research.google.com/) |
| Data Analysis        | `pandas`, `numpy` |
| Visualization        | `matplotlib`, `seaborn` |
| Modeling             | `scikit-learn` |
| LLM Interface        | [LangChain](https://python.langchain.com/) |
| LLM Hosting & API    | [Replicate](https://replicate.com/) |
| LLM Provider         | [IBM Granite Foundation Models](https://research.ibm.com/interactive/granite/) |

---

## 🤖 LLM Agent Integration

We use `LangChain`'s experimental `create_pandas_dataframe_agent` to build a smart agent capable of responding to human language questions.

To enhance the response quality, we integrate **IBM Granite 13B Chat** model via Replicate.

### 🔐 Using Replicate API (IBM Granite)

To use the IBM Granite model:

1. Create an account on [replicate.com](https://replicate.com/)
2. Go to your profile → API Token
3. Paste this token in your Colab notebook:

```python
import os
os.environ["REPLICATE_API_TOKEN"] = "your_token_here"
```

4. Run the model via:
```python
import replicate

response = replicate.run(
  "ibm/granite-13b-chat",
  input={"prompt": "Summarize the most extroverted traits in this dataset"}
)
```

## 📊 Example Outputs
Some questions this project can answer:

- “What is the most frequent personality type?”
- “Which features are strongly linked to extroversion?”
- “How many Ambiverts are in the dataset?”
- “What’s the average emotional score per personality?”

_(Tip: You can ask your own questions using the LangChain agent interface inside the notebook!)_

---

🧪 How to Use This Project
- Open the notebook in Google Colab
- Upload the file: personality_synthetic_dataset.csv
- Set your Replicate API Token to access IBM LLMs
- Run all the cells sequentially
- Try your own natural language queries via the LangChain agent

🚀 Key Learning Outcomes
- 🌱 Apply machine learning to psychology-related data
- 🛠 Build AI agents that interact with structured data using LLMs
- 🧠 Use LangChain to wrap pandas DataFrames with language models
- 🔗 Access and integrate IBM LLMs via Replicate
- 💡 Gain real-world exposure to AI workflow integration

---

💡 Real-World Applications
- 🧬 Educational tools for personality understanding
- 📊 HR analytics and organizational behavior
- 🤖 AI-powered chatbots for psychological profiling
- 📚 Research simulation for behavioral science

---

📌 Future Enhancements
- Integrate with real-world surveys (e.g., MBTI, Big Five)
- Build interactive web UI using Gradio or Streamlit
- Add visual-to-text agents that describe charts using LLMs
- Use fine-tuned models for more domain-specific insight

---

📄 License
This project is provided for educational and non-commercial use only.
All datasets and APIs used belong to their respective providers under their own licenses.

---

🙌 Acknowledgments
- 🎓 Hacktiv8 Indonesia (educational collaboration)
- 🤝 IBM Granite Foundation Models
- 🔗 Replicate for hosting LLMs
- 📊 Kaggle Dataset by Miadul
- 🧠 LangChain Python API

✨ _Created with heart by **Ketsar** — Exploring data, psychology, and AI through practical experimentation._
