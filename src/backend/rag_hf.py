import os
import logging
from typing import Union, List, Dict, Any
from pathlib import Path

import torch
from dotenv import load_dotenv

from langchain.chains import RetrievalQA
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

# ========== 設定読み込み ========== #
load_dotenv()

base_dir = Path(__file__).resolve().parent.parent  # src/backend から2つ上に戻る
vector_store_path = base_dir / os.getenv("FAISS_DB_DIR")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

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
        from langchain_community.llms import HuggingFacePipeline

        logging.info(f"🤖 HuggingFace LLM（{llm_model_name}）をロード中...")
        tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
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

    if vector_store_path is None:
        vector_store_path = FAISS_DB_DIR

    # ベクトルDBの存在確認
    if not (Path(vector_store_path) / "index.faiss").exists():
        raise FileNotFoundError(f"❌ index.faiss が見つかりません: {vector_store_path}")
    if not (Path(vector_store_path) / "index.pkl").exists():
        raise FileNotFoundError(f"❌ index.pkl が見つかりません: {vector_store_path}")

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
        logging.error("❌ 回答生成中にエラー: %s", str(e))
        return {
            "answer": "⚠️ 回答生成中にエラーが発生しました。",
            "source_documents": []
        }

    # 出力整形
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


# ========== CLI 実行 ========== #
if __name__ == "__main__":
    query = "不法行為による損害賠償とは？"
    result = rag_chatbot(query, use_openai=True)
    print("💬 回答:\n", result["answer"])
    print("📚 参照文書:")
    for i, doc in enumerate(result["source_documents"], 1):
        metadata = doc.get("metadata", {})
        print(f"{i}. {metadata.get('source', 'No Source')}")
        print(doc["page_content"][:200], "...\n")

