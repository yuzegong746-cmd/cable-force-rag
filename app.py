"""
钢索轴力智能问答系统 - Web 界面
运行：streamlit run app.py
"""

import os
import sys
import streamlit as st

st.set_page_config(
    page_title="钢索轴力智能问答系统",
    page_icon="🔩",
    layout="wide"
)

@st.cache_resource(show_spinner="正在加载模型和知识库，首次约需1分钟...")
def load_rag_system(api_key: str):
    from rag_pipeline import CableRAGSystem
    return CableRAGSystem(openai_api_key=api_key)


with st.sidebar:
    st.title("⚙️ 系统配置")

    api_key = st.text_input(
        "Google API Key",
        type="password",
        value=os.environ.get("GOOGLE_API_KEY", ""),
        help="输入您的 Google Gemini API Key"
    )

    st.divider()
    st.markdown("### 📌 示例问题")
    example_questions = [
        "温度60度、张力500N时轴力是多少？",
        "Physics-Informed RBFNN 比普通MLP有什么优势？",
        "热膨胀效应如何影响钢索轴力？",
        "模型的预测精度指标是多少？",
        "什么是热-力耦合效应？",
    ]
    for q in example_questions:
        if st.button(q, use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    st.markdown("### 📂 上传文档")
    uploaded = st.file_uploader(
        "上传实验报告（PDF/TXT）",
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    if uploaded:
        os.makedirs("data", exist_ok=True)
        for f in uploaded:
            with open(f"data/{f.name}", "wb") as out:
                out.write(f.read())
        st.success(f"已上传 {len(uploaded)} 个文件，重启系统生效")

    if st.button("🔄 重建知识库", use_container_width=True):
        st.cache_resource.clear()
        st.rerun()


st.title("🔩 钢索轴力智能问答系统")
st.caption("基于 RAG + Physics-Informed RBFNN | Cable Force Intelligent QA")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("prediction"):
            pred = msg["prediction"]
            col1, col2, col3 = st.columns(3)
            col1.metric("预测轴力", f"{pred['predicted_force']:.2f} kN")
            col2.metric("置信下界", f"{pred['lower_bound']:.2f} kN")
            col3.metric("置信上界", f"{pred['upper_bound']:.2f} kN")
        if msg.get("sources"):
            with st.expander("📖 参考来源"):
                for s in msg["sources"]:
                    st.markdown(f"**{s['source']}**\n\n{s['content']}...")

prefill = st.session_state.pop("prefill", "")
question = st.chat_input("输入您的问题，例如：温度60度、张力500N时轴力是多少？") or prefill

if question:
    if not api_key:
        st.error("请在左侧侧边栏输入 Google API Key")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                rag = load_rag_system(api_key)
                result = rag.query(question)

                st.markdown(result["answer"])

                if result.get("prediction"):
                    pred = result["prediction"]
                    st.divider()
                    col1, col2, col3 = st.columns(3)
                    col1.metric("预测轴力", f"{pred['predicted_force']:.2f} kN")
                    col2.metric("置信下界", f"{pred['lower_bound']:.2f} kN")
                    col3.metric("置信上界", f"{pred['upper_bound']:.2f} kN")

                if result.get("retrieved_docs"):
                    with st.expander("📖 参考来源"):
                        for doc in result["retrieved_docs"]:
                            st.markdown(f"**{doc['source']}**\n\n{doc['content']}...")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "prediction": result.get("prediction"),
                    "sources": result.get("retrieved_docs", [])
                })

            except Exception as e:
                err_msg = f"❌ 出错：{str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({"role": "assistant", "content": err_msg})
