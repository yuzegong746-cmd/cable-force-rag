# 钢索轴力智能问答系统

**Cable Force Intelligent QA System based on RAG + Physics-Informed Neural Networks**

[!\[Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[!\[LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com)
[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 项目简介

本项目将**6种钢索轴力神经网络预测模型**与\*\*大语言模型（LLM）\*\*结合，构建了一个面向结构工程师的智能问答系统。

基于 RWTH Aachen 大学实习项目，对比了 MLP、CNN、RBFNN、ModularRBFNN、PI-ModularRBFNN、ProductNN 六种架构，最优模型 R²=0.9991。

用户可以用自然语言：

* 查询钢索在特定工况下的轴力预测值（自动调用神经网络推理）
* 询问模型原理、热-力耦合机制等专业问题
* 上传实验报告，系统自动构建 RAG 知识库并回答问题

## 核心技术

|模块|技术|说明|
|-|-|-|
|知识库构建|LangChain + FAISS|文档切片 → Embedding → 向量存储|
|语义检索|sentence-transformers|多语言语义相似度检索|
|轴力预测|PI-ModularRBFNN / MLP|物理约束正则化神经网络，R²=0.9991|
|对话生成|Google Gemini 2.0 Flash|结合检索内容生成专业回答|
|Web 界面|Streamlit|快速部署交互界面|

## 模型精度

|模型|R²|参数量|推理时间|
|-|-|-|-|
|PI-ModularRBFNN|**0.9991**|99K|\~1μs|
|MLP \[256,128,64]|**0.9991**|86K|\~0.8μs|
|ModularRBFNN|0.9982|99K|\~1μs|
|RBFNN|0.9977|**513**|**0.24μs**|
|CNN|0.9975|39K|\~0.5μs|
|ProductNN|0.9968|18K|\~0.3μs|

## 项目结构

```
cable-rag/

├── models/                          # 训练好的模型权重

│   ├── pi\_modularRBFNN\_seed42\_best.pth   # 最优：物理约束RBFNN

│   ├── mlp\_seed42\_best.pth               # 最优：多层感知机

│   ├── modularRBFNN\_seed42\_best.pth

│   ├── rbf\_nn\_seed42\_best.pth

│   ├── product\_nn\_seed42\_best.pth

│   └── cnn\_seed42\_best.pth

├── predictor.py                     # 神经网络推理模块

├── rag\_pipeline.py                  # RAG 问答核心逻辑

├── app.py                           # Streamlit Web 界面

└── requirements.txt```

## 快速开始

### 1\. 克隆项目

```bash
git clone https://github.com/yuzegong746-cmd/cable-force-rag.git
cd cable-force-rag
```

### 2\. 安装依赖

```bash
pip install -r requirements.txt
```

### 3\. 配置 Google API Key

前往 [Google AI Studio](https://aistudio.google.com) 免费申请 Key。

```bash
# Windows PowerShell
$env:GOOGLE\_API\_KEY="你的key"

# Linux/Mac
export GOOGLE\_API\_KEY="你的key"
```

### 4\. 启动系统

```bash
streamlit run app.py
```

浏览器访问 http://localhost:8501

## 使用示例

**预测轴力：**

> 位移0.015m、温度600K时轴力是多少？

**专业问答：**

> PI-ModularRBFNN 相比普通 RBFNN 有什么优势？

> 温度升高如何影响塑性硬化模量？

## 物理背景

* **缆索参数**：L=10m，r=0.03m，E=210GPa，屈服应力k=252MPa
* **输入范围**：位移 w∈\[0, 0.025]m，温度 T∈\[273, 1000]K
* **输出**：钢索轴力 F∈\[0, \~1155kN]
* **本构模型**：双线性弹塑性 + 温度相关塑性硬化模量 H(T)

## License

MIT License

