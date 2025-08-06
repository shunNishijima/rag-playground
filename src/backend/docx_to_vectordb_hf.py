import os
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from docx import Document as DocxDocument
from langchain_core.documents import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from tqdm import tqdm
import fitz  # PyMuPDF
from pdf2image import convert_from_path
import pytesseract
import re

# ============================
# .env 読み込み
# ============================
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================
# 切り替え可能な埋め込みモデル
# ============================
USE_OPENAI = True  # ← 切り替え可能

if USE_OPENAI:
    from langchain_openai import OpenAIEmbeddings
    if not OPENAI_API_KEY:
        raise ValueError("❌ OPENAI_API_KEY が .env に設定されていません。")
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    VECTOR_STORE_PATH = Path("vectorstore")
    print("🧠 OpenAI埋め込みモデルを使用します。")
else:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
    VECTOR_STORE_PATH = Path("vectorstore_hf")
    print("🧠 HuggingFace埋め込みモデルを使用します。")

# ============================
# パス設定
# ============================
DOCX_DIR = Path("data/docx")
PDF_DIR = Path("data/pdf")
TXT_DIR = Path("data/all")

# ============================
# OCRあり or なしに関わらずPDF処理
# ============================
def process_and_save_with_ocr():
    documents = []

    for pdf_path in Path(PDF_DIR).glob("*.pdf"):
        print(f"📄 PDF処理中: {pdf_path.name}")

        # スタイル付き抽出（バツ印や赤文字優先）
        pages = extract_text_with_styles(pdf_path)

        # ページが空ならOCRへフォールバック
        if all(not p.strip() for p in pages):
            print(f"⚠️ スタイル抽出失敗: {pdf_path.name} → OCRに切り替え")
            pages = extract_text_from_scanned_pdf(pdf_path)

        # OCRでも空ならテキストPDFとして処理
        if all(not p.strip() for p in pages):
            print(f"⚠️ OCR失敗: {pdf_path.name} → テキスト型PDFとして処理します")
            full_text = extract_text_from_pdf(pdf_path)
            if not full_text.strip():
                print(f"❌ PDFテキスト抽出も失敗: {pdf_path.name} → スキップ")
                continue
            metadata = {
                "source": pdf_path.name,
                "id": f"{pdf_path.stem}",
                "page_number": 1,
                "section": extract_section_title(full_text),
            }
            documents.append(LangchainDocument(page_content=full_text, metadata=metadata))
        else:
            for i, page_text in enumerate(pages):
                if not page_text.strip():
                    continue
                metadata = {
                    "source": pdf_path.name,
                    "id": f"{pdf_path.stem}_p{i+1}",
                    "page_number": i + 1,
                    "section": extract_section_title(page_text),
                }
                documents.append(LangchainDocument(page_content=page_text, metadata=metadata))

    if not documents:
        print("❌ 文書が1つも生成できませんでした。")
        return

    print(f"🧱 チャンク分割中...（文書数: {len(documents)}）")
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=512, chunk_overlap=128)
    split_docs = splitter.split_documents(documents)
    print(f"✅ 分割後チャンク数: {len(split_docs)}")

    save_vector_store(split_docs, VECTOR_STORE_PATH)


# ============================
# スタイルを保持しながらOCR
# ============================
def extract_text_with_styles(pdf_path: Path) -> List[str]:
    """PDFページごとに、色や装飾を判別しつつテキスト抽出"""
    doc = fitz.open(str(pdf_path))
    processed_pages = []

    for page in doc:
        page_text = ""
        blocks = page.get_text("dict")["blocks"]
        for b in blocks:
            if "lines" not in b:
                continue
            for line in b["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    # 判別条件
                    color = span.get("color", 0)
                    is_strike = span.get("flags", 0) & 8 != 0  # 8 = strike-through

                    # ❌ 打ち消し線の文字は無視
                    if is_strike:
                        continue

                    # ✅ 色付き文字（赤・緑）のみ抽出（例：RGB値）
                    if color in [0xFF0000, 0x00FF00]:  # 赤または緑
                        page_text += f"{text} "

                    # ⚪ 通常文字も抽出（優先度低）
                    elif color == 0x000000:
                        page_text += f"{text} "

        processed_pages.append(page_text.strip())

    return processed_pages


# ============================
# テキスト型PDF抽出
# ============================
def extract_text_from_pdf(pdf_path: Path) -> str:
    try:
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        print(f"❌ PDF読み取り失敗: {pdf_path.name} - {e}")
        return ""

# ============================
# OCRでページごとに抽出
# ============================
def extract_text_from_scanned_pdf(pdf_path: Path) -> List[str]:
    try:
        images = convert_from_path(str(pdf_path))
        return [pytesseract.image_to_string(img, lang="jpn") for img in images]
    except Exception as e:
        print(f"❌ OCR失敗: {pdf_path.name} - {e}")
        return []

# ============================
# 章タイトル抽出
# ============================
def extract_section_title(text: str) -> str:
    """章タイトルらしきものを抽出"""
    patterns = [
        r"(第[0-9一二三四五六七八九十]+章\s*.+)",
        r"^\s*[0-9]+\.\s+.+",  # 1. はじめに
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.MULTILINE)
        if match:
            try:
                # グループが存在する場合のみ
                return match.group(1).strip()
            except IndexError:
                return match.group(0).strip()
    return "不明"


# ============================
# FAISSベクトル保存
# ============================
def save_vector_store(docs: List[LangchainDocument], output_dir: Path, batch_size: int = 50):
    if not docs:
        print("⚠️ 文書が空です")
        return

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    print(f"🧠 埋め込み開始: {len(texts)}件")
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="🔄 ベクトル化中"):
        batch = texts[i:i + batch_size]
        try:
            batch_vectors = embedding_model.embed_documents(batch)
            embeddings.extend(batch_vectors)
        except Exception as e:
            print(f"❌ 埋め込み失敗（{i}件目）: {e}")

    if not embeddings:
        print("❌ ベクトルが生成されませんでした。")
        return

    try:
        faiss_db = FAISS.from_embeddings(
            list(zip(texts, embeddings)),
            embedding=embedding_model,
            metadatas=metadatas
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss_db.save_local(str(output_dir))
        print(f"✅ ベクトル保存完了: {output_dir.resolve()}")
    except Exception as e:
        print(f"❌ FAISS 保存失敗: {e}")

# ============================
# 実行フロー
# ============================
if __name__ == "__main__":
    process_and_save_with_ocr()
