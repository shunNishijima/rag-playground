import os
import logging
from typing import Union, List, Dict, Any
from pathlib import Path
import traceback
import torch
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
try:
    import streamlit as st
    secrets = st.secrets
except ModuleNotFoundError:
    secrets = {}


# ========== 設定読み込み ========== #
load_dotenv()

def get_secret(key, default=None):
    if "st" in globals() and hasattr(st, "secrets"):
        return os.getenv(key) or st.secrets.get(key, default)
    return os.getenv(key, default)


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
base_dir = Path(__file__).resolve().parents[2]  # frontend → src → project root
def get_vector_store_path(use_openai: bool) -> Path:
    """use_openaiに応じてベクトルストアのパスを切り替える"""
    store_dir_name = "vectorstore" if use_openai else "vectorstore_hf"
    return base_dir / store_dir_name


# ========== ログ設定 ========== #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# ========== 使用デバイス確認 ========== #
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"✅ 使用デバイス: {device}")

# ========== モデル名 ========== #
embedding_model_name = "intfloat/multilingual-e5-small"
llm_model_name = "rinna/youri-7b-chat"

# ========== 埋め込みモデル選択 ========== #
def get_embedding_model(use_openai: bool):
    if use_openai:
        if not OPENAI_API_KEY:
            raise EnvironmentError("❌ OPENAI_API_KEY が未設定です。")
        from langchain_openai import OpenAIEmbeddings
        logging.info("🧠 OpenAI Embeddings を使用します。")
        return OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logging.info("🧠 HuggingFace Embeddings を使用します。")
        return HuggingFaceEmbeddings(model_name=embedding_model_name)

# ========== LLM モデル選択 ========== #
def get_llm(use_openai: bool):
    if use_openai:
        from langchain_openai import ChatOpenAI
        logging.info("🤖 OpenAI LLM（gpt-4o）を使用します。")
        return ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        from langchain_huggingface import HuggingFacePipeline  # ✅ 修正ポイント

        logging.info(f"🤖 HuggingFace LLM（{llm_model_name}）をロード中...")
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(
            llm_model_name,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            return_full_text=False,
            max_new_tokens=512,
            do_sample=False
        )

        logging.info("✅ HuggingFace LLM のロード完了")
        return HuggingFacePipeline(pipeline=pipe)


# ========== チェーン生成 ========== #
def get_retrieval_chain(llm, vectorstore: FAISS) -> RetrievalQA:
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

# ========== RAG チャット関数 ========== #
def rag_chatbot(
    input_text: str,
    vector_store_path: str = None,
    use_openai: bool = False
) -> Dict[str, Union[str, List[Dict[str, Any]]]]:

    # ✅ vectorstoreパスを use_openai に応じて自動切り替え
    if vector_store_path is None:
        vector_store_path = get_vector_store_path(use_openai)
    else:
        vector_store_path = Path(vector_store_path)


    # ✅ パスが存在しなければ詳細エラー表示
    index_faiss = vector_store_path / "index.faiss"
    index_pkl = vector_store_path / "index.pkl"

    if not index_faiss.exists():
        raise FileNotFoundError(
            f"❌ index.faiss が見つかりません: {index_faiss}\n"
            f"🛠️ 対応策: vectorstore構築スクリプトを実行してください。"
        )
    if not index_pkl.exists():
        raise FileNotFoundError(
            f"❌ index.pkl が見つかりません: {index_pkl}\n"
            f"🛠️ 対応策: vectorstore構築スクリプトを実行してください。"
        )

    # モデルロード
    embedding_model = get_embedding_model(use_openai)
    llm = get_llm(use_openai)

    logging.info(f"📂 ベクトルDBを読み込み: {vector_store_path}")
    faiss_db = FAISS.load_local(
        vector_store_path,
        embedding_model,
        allow_dangerous_deserialization=True
    )

    # チェーン作成と実行
    retrieval_chain = get_retrieval_chain(llm, faiss_db)

    logging.info(f"💬 ユーザー入力: {input_text}")
    try:
        result = retrieval_chain.invoke({"query": input_text})
    except Exception as e:
        logging.error("❌ 回答生成中にエラー:\n%s", traceback.format_exc())
        return {
            "answer": "⚠️ 回答生成中にエラーが発生しました。",
            "source_documents": []
        }

    logging.info("✅ 回答生成完了")
    return {
        "answer": result["result"],
        "source_documents": [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }

# ========== RAG チャットストリーミング関数 ========== #
def rag_chatbot_stream(input_text: str, vector_store_path: str = None, use_openai: bool = False):
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableMap

    # パス設定
    if vector_store_path is None:
        vector_store_path = get_vector_store_path(use_openai)
    else:
        vector_store_path = Path(vector_store_path)

    index_faiss = vector_store_path / "index.faiss"
    index_pkl = vector_store_path / "index.pkl"
    if not index_faiss.exists() or not index_pkl.exists():
        yield "⚠️ vectorstore が存在しません。"
        return

    # ベクトルストア読み込み
    embedding_model = get_embedding_model(use_openai)
    llm = get_llm(use_openai)
    faiss_db = FAISS.load_local(vector_store_path, embedding_model, allow_dangerous_deserialization=True)
    retriever = faiss_db.as_retriever(search_kwargs={"k": 3})

    try:
        if use_openai:
            # ✅ OpenAI LLM（Streaming対応）
            from langchain_openai import ChatOpenAI
            from langchain_core.messages import HumanMessage

            docs = retriever.invoke(input_text)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""以下の文書を参考に、質問に答えてください。

            文書:
            {context}

            質問: {input_text}
            """

            # ChatOpenAIはstream=Trueに設定
            stream_llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                streaming=True,
                openai_api_key=OPENAI_API_KEY
            )

            # 🔽 HumanMessage をリストにするのが重要！
            for chunk in stream_llm.stream([HumanMessage(content=prompt)]):
                yield chunk.content

            yield "\n\n[END]"

        else:
            # ✅ HuggingFace LLM（Streaming対応）
            from transformers import TextStreamer
            tokenizer = llm.pipeline.tokenizer
            model = llm.pipeline.model

            docs = retriever.invoke(input_text)
            context = "\n\n".join([doc.page_content for doc in docs])
            prompt = f"""以下の文書を参考に、質問に答えてください。

文書:
{context}

質問: {input_text}
"""

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            import io
            from contextlib import redirect_stdout
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                model.generate(**inputs, max_new_tokens=512, streamer=streamer)

            # 文字単位でストリーミング出力
            for char in buffer.getvalue():
                yield char
            yield "\n\n[END]"

    except Exception as e:
        logging.error("❌ ストリーミング生成中にエラー:\n%s", traceback.format_exc())
        yield f"⚠️ エラーが発生しました: {e}"


# ========== CLI 実行 ========== #
# if __name__ == "__main__":
#     query = "取締役等に関する規律の見直しについてどのようなものがあったか教えてください。"
#     result = rag_chatbot(query, use_openai=True)
#     print("💬 回答:\n", result["answer"])
#     print("📚 参照文書:")
#     for i, doc in enumerate(result["source_documents"], 1):
#         metadata = doc.get("metadata", {})
#         print(f"{i}. {metadata.get('source', 'No Source')}")
#         print(doc["page_content"][:200], "...\n")

if __name__ == "__main__":
    query = "取締役等に関する規律の見直しについてどのようなものがあったか教えてください。"
    for token in rag_chatbot_stream(query, use_openai=True):  # or False
        print(token, end="", flush=True)
