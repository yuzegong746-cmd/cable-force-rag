"""
钢索轴力智能问答系统 - RAG Pipeline
"""

import os
import json
import re
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.documents import Document

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_google_genai import ChatGoogleGenerativeAI
from predictor import CableForcePredictor


def _get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )


def build_knowledge_base(docs_dir: str = "data/", save_path: str = "faiss_index"):
    print("📚 正在构建知识库...")
    documents = []

    if os.path.exists(docs_dir):
        for filename in os.listdir(docs_dir):
            filepath = os.path.join(docs_dir, filename)
            if filename.endswith(".pdf"):
                try:
                    loader = PyPDFLoader(filepath)
                    documents.extend(loader.load())
                    print(f"  ✅ 加载 PDF: {filename}")
                except Exception as e:
                    print(f"  ⚠️  {filename}: {e}")
            elif filename.endswith((".txt", ".tex")):
                try:
                    loader = TextLoader(filepath, encoding="utf-8")
                    documents.extend(loader.load())
                    print(f"  ✅ 加载文本: {filename}")
                except Exception as e:
                    print(f"  ⚠️  {filename}: {e}")

    if not documents:
        print("  ⚠️  data/ 为空，使用内置示例文档")
        documents = _load_sample_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=50,
        separators=["\n\n", "\n", "。", "；", " "]
    )
    chunks = splitter.split_documents(documents)
    print(f"  📄 共 {len(chunks)} 个文本块")

    print("  🔢 生成向量中（首次约需1分钟）...")
    vectorstore = FAISS.from_documents(chunks, _get_embeddings())
    vectorstore.save_local(save_path)
    print(f"  💾 已保存至 {save_path}/\n")
    return vectorstore


def load_knowledge_base(save_path: str = "faiss_index"):
    vectorstore = FAISS.load_local(
        save_path, _get_embeddings(), allow_dangerous_deserialization=True
    )
    print(f"✅ 知识库已加载（{save_path}/）")
    return vectorstore


def extract_params(question: str) -> Optional[dict]:
    params = {}

    disp_match = re.search(r'位移[为是]?\s*(\d+\.?\d*)\s*m?', question)
    if disp_match:
        params['displacement'] = float(disp_match.group(1))

    temp_match = re.search(r'温度[为是]?\s*(\d+\.?\d*)\s*(K|℃|度|°C)?', question)
    if temp_match:
        val = float(temp_match.group(1))
        unit = temp_match.group(2) or ''
        params['temperature'] = val if (unit == 'K' or val >= 200) else val + 273.15

    tension_match = re.search(r'[预张]?力[为是]?\s*(\d+\.?\d*)\s*N', question)
    if tension_match:
        params['tension'] = float(tension_match.group(1))

    if params.get('temperature') or params.get('displacement'):
        return params
    return None


class CableRAGSystem:
    def __init__(self, openai_api_key: str = "", index_path: str = "faiss_index"):
        self.predictor = CableForcePredictor()

        if os.path.exists(index_path):
            self.vectorstore = load_knowledge_base(index_path)
        else:
            self.vectorstore = build_knowledge_base(save_path=index_path)

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            google_api_key=openai_api_key,
        )

        self.system_prompt = """你是结构工程与机器学习专家，专注于钢索轴力分析与预测。
项目背景：RWTH Aachen 实习项目，对比6种神经网络（MLP/CNN/RBFNN/ModularRBFNN/PI-ModularRBFNN/ProductNN）。
输入：位移w∈[0,0.025]m，温度T∈[273,1000]K；输出：轴力F（kN）。最优R²=0.9991。
回答要求：结合参考资料，专业准确，有物理解释，不确定内容说明局限性。"""

    def query(self, question: str, top_k: int = 4) -> dict:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": top_k})
        relevant_docs = retriever.invoke(question)

        context = "\n\n---\n\n".join([
            f"[来源: {doc.metadata.get('source', '知识库')}]\n{doc.page_content}"
            for doc in relevant_docs
        ])

        prediction_result = None
        params = extract_params(question)
        if params:
            try:
                prediction_result = self.predictor.predict(params)
            except Exception as e:
                print(f"  ⚠️  预测失败：{e}")

        user_content = f"【参考资料】\n{context}\n\n"
        if prediction_result:
            w = prediction_result['input_physical']['w_m']
            T = prediction_result['input_physical']['T_K']
            user_content += (
                f"【神经网络预测结果】\n"
                f"工况：w={w:.4f}m，T={T:.1f}K\n"
                f"预测轴力：{prediction_result['predicted_force_kN']:.2f} kN\n"
                f"置信区间：{prediction_result['lower_bound_kN']:.2f}~{prediction_result['upper_bound_kN']:.2f} kN\n"
                f"模型：{prediction_result['method']}\n\n"
            )
        user_content += f"【问题】\n{question}"

        response = self.llm.invoke([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_content)
        ])

        # 兼容 app.py 的字段名
        if prediction_result:
            prediction_result['predicted_force'] = prediction_result['predicted_force_kN']
            prediction_result['lower_bound'] = prediction_result['lower_bound_kN']
            prediction_result['upper_bound'] = prediction_result['upper_bound_kN']

        return {
            "question": question,
            "answer": response.content,
            "retrieved_docs": [
                {"source": d.metadata.get("source", ""), "content": d.page_content[:200]}
                for d in relevant_docs
            ],
            "prediction": prediction_result,
            "params_extracted": params
        }


def _load_sample_documents():
    sample_texts = [
        """钢索轴力预测研究背景（RWTH Aachen 实习项目）
对比6种神经网络：MLP、ProductNN、RBFNN、CNN、ModularRBFNN、PI-ModularRBFNN。
训练数据：5000样本，Stratified LHS采样，seed=42。
输入：位移w∈[0,0.025]m，温度T∈[273,1000]K；输出：轴力F（kN）。
最优：MLP和PI-ModularRBFNN，R²=0.9991，RMSE≈6.81kN。""",

        """物理模型：双线性弹塑性本构
参数：L=10m，r=0.03m，A=π×0.03²≈2.83×10⁻³m²，E=210GPa，k=252MPa，ε_p=0.0012
塑性硬化模量：H(T)=H₀-B₁·T·exp(-T₀/T)，H₀=120.64GPa，B₁=161MPa/K，T₀=1500K
轴力：F=σ×A；弹性段σ=E·ε；塑性段σ=k+H(T)·(ε-ε_p)
输出范围：F∈[0,~1155kN]（最大值在w=0.025m,T=273K）""",

        """模型精度对比
PI-ModularRBFNN：R²=0.9991，参数量99K，物理约束权重λ=0.2
MLP [256,128,64]：R²=0.9991，参数量86K，LeakyReLU+BatchNorm
RBFNN（128中心）：R²=0.9977，仅513参数，推理≈0.24μs，参数效率最高
ModularRBFNN：R²=0.9982，维度解耦设计
CNN（1D卷积）：R²=0.9975，39K参数
ProductNN：R²=0.9968，乘积单元网络""",

        """温度效应与归一化说明
温度升高→H(T)下降（273K时119.64GPa→1000K时84.07GPa，降幅30%）
塑性段：温度越高，相同位移下轴力越小
归一化：w_norm=w/0.025，T_norm=(T-273)/727，F_norm=F/1155281.4N
推理速度：神经网络μs级，远快于FEM的45分钟""",
    ]
    return [Document(page_content=t, metadata={"source": "内置知识库"}) for t in sample_texts]


if __name__ == "__main__":
    import sys
    api_key = os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        print("❌ 请设置 $env:GOOGLE_API_KEY='你的key'")
        sys.exit(1)
    rag = CableRAGSystem(openai_api_key=api_key)
    while True:
        q = input("你：").strip()
        if q.lower() in ["quit", "exit", "退出"]:
            break
        if q:
            print(f"\n助手：{rag.query(q)['answer']}\n")
