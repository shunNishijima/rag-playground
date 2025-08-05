# ============================
# streamlitを使ったフロントエンド（模範回答対応版）
# ============================
import time
import pathlib
import sys
import json
from datetime import datetime
import streamlit as st

# プロジェクトルートの `src` ディレクトリをパスに追加
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from backend.rag_hf import rag_chatbot

# ========== ログファイル定義 ========== #
LOG_FILE = pathlib.Path("logs/ragas_eval_log.jsonl")
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# ========== 参照回答（模範解答）読み込み ========== #
REFERENCE_FILE = pathlib.Path("data/reference.json")
if REFERENCE_FILE.exists():
    with REFERENCE_FILE.open("r", encoding="utf-8") as f:
        reference_data = json.load(f)
    reference_dict = {item["question"]: item["reference"] for item in reference_data}
else:
    reference_dict = {}

# ========== RAGAS評価用ログ関数 ========== #
def log_ragas_sample(question: str, response: dict, reference: str):
    sample = {
        "question": question,
        "answer": response["answer"],
        "contexts": [doc["page_content"] for doc in response["source_documents"]],
        "reference": reference,
        "timestamp": datetime.now().isoformat()
    }

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(sample, ensure_ascii=False) + "\n")

# ========== Streamlit UI定義 ========== #
def render():
    st.set_page_config(page_title="法律相談チャットボット", page_icon="⚖️")

    st.sidebar.header("設定")
    use_openai = st.sidebar.radio(
        "埋め込みモデルを選択",
        options=["HuggingFace", "OpenAI"],
        index=0,
        help="埋め込みベクトル生成に使用するモデル"
    )
    use_openai = (use_openai == "OpenAI")

    st.title("⚖️ 法律相談チャットボット")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # 過去のメッセージ表示
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)

    # ユーザー入力受付
    if prompt := st.chat_input("ここに法律相談の質問を入力してください"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("回答を生成中..."):
                response = rag_chatbot(prompt, use_openai=use_openai)
                result = response["answer"]
                source_documents = response["source_documents"]

            # 🔽 参照回答（reference）取得
            reference = reference_dict.get(prompt, "（模範解答が見つかりません）")

            # 🔽 RAGAS ログ記録
            log_ragas_sample(prompt, response, reference)

            # ソース整形
            details_message = ""
            if source_documents:
                details_message += "<br><span style='font-size: small; color: gray;'>\n参考文書:</span><ul>"
                for doc in source_documents:
                    metadata = doc.get("metadata", {})
                    source_path = metadata.get("source") or metadata.get("id") or "不明"
                    title = pathlib.Path(source_path).name
                    details_message += f"<li><b>{title}</b></li>"
                details_message += "</ul>"
            else:
                details_message += "<span style='font-size: small; color: gray;'>参考資料はありません。</span>"

            # 回答表示
            answer = f"{result}<br><br><b>📑 模範解答:</b> {reference}<br>{details_message}"
            st.markdown(answer, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()

# ========== 実行 ========== #
if __name__ == "__main__":
    render()
