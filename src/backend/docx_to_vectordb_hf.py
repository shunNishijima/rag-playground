import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm

# ============================
# .env 読み込み
# ============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================
# 切り替え可能な埋め込みモデル
# ============================
USE_OPENAI = False  # ← OpenAI Embedding を使うかどうか

if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY が .env に設定されていません。")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    print("🧠 OpenAI埋め込みモデルを使用します。")
else:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    print("🧠 HuggingFace埋め込みモデルを使用します。")

# ============================
# パス設定
# ============================
DOCX_DIR = Path("data/docx")
TXT_DIR = Path("data/all")
if USE_OPENAI:
    FAISS_DB_DIR = Path("vectorstore")
else:
    FAISS_DB_DIR = Path("vectorstore_hf")
# ============================
# DOCX → TXT 変換関数
# ============================
def convert_docx_directory(docx_dir: Path, txt_output_dir: Path):
    txt_output_dir.mkdir(parents=True, exist_ok=True)
    for docx_file in docx_dir.glob("*.docx"):
        txt_file = txt_output_dir / f"{docx_file.stem}.txt"
        if txt_file.exists():
            print(f"⏭️ スキップ: {txt_file.name}")
            continue
        try:
            doc = DocxDocument(docx_file)
            text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
            txt_file.write_text(text, encoding="utf-8")
            print(f"✅ 変換: {docx_file.name}")
        except Exception as e:
            print(f"❌ エラー: {docx_file.name} - {e}")

# ============================
# TXT → LangchainDocument 読み込み
# ============================
def load_txt_directory(txt_dir: Path) -> List[LangchainDocument]:
    docs = []
    for file in txt_dir.glob("*.txt"):
        try:
            text = file.read_text(encoding="utf-8")
            if text.strip():
                docs.append(LangchainDocument(
                    page_content=text,
                    metadata={
                        "source": file.name,
                        "id": file.stem  # これを追加
                    }
                ))
                print(f"📄 ロード: {file.name}")
        except Exception as e:
            print(f"⚠️ 読み込み失敗: {file.name} - {e}")
    return docs


# ============================
# チャンク分割（metadata付き）
# ============================
def split_documents(docs: List[LangchainDocument], chunk_size=1000, chunk_overlap=100) -> List[LangchainDocument]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = splitter.split_documents(docs)
    return chunked_docs

# ============================
# ベクトル保存
# ============================
def save_vector_store(docs: List[LangchainDocument], output_dir: Path, batch_size: int = 50):
    if not docs:
        print("⚠️ 文書が空です")
        return

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    print(f"🔢 ベクトル変換: {len(texts)}件, 最長: {max(len(t) for t in texts)}文字")

    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔄 埋め込み実行中"):
        batch = texts[i:i + batch_size]
        try:
            batch_vectors = embedding_model.embed_documents(batch)
            embeddings.extend(batch_vectors)
        except Exception as e:
            print(f"❌ 埋め込み失敗（{i}件目）: {e}")

    if not embeddings:
        print("❌ ベクトルが空です。保存できません。")
        return

    try:
        faiss_db = FAISS.from_embeddings(
            list(zip(texts, embeddings)),
            embedding=embedding_model,
            metadatas=metadatas
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss_db.save_local(str(output_dir))
        print(f"✅ 保存完了: {output_dir.resolve()}")
    except Exception as e:
        print(f"❌ FAISS 保存失敗: {e}")

# ============================
# 実行フロー
# ============================
def process_and_save():
    print(f"📂 DOCX → TXT 変換: {DOCX_DIR.resolve()}")
    convert_docx_directory(DOCX_DIR, TXT_DIR)

    print(f"📁 TXT読込: {TXT_DIR.resolve()}")
    docs = load_txt_directory(TXT_DIR)
    print(f"✅ 文書数: {len(docs)}")

    print("✂️ チャンク分割中...")
    chunked_docs = split_documents(docs)
    print(f"✅ チャンク数: {len(chunked_docs)}")

    print("💾 ベクトル保存中...")
    save_vector_store(chunked_docs, FAISS_DB_DIR)

if __name__ == "__main__":
    process_and_save()
