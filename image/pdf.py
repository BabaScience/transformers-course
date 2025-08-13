from transformers import pipeline
from pathlib import Path
from PIL import Image
import fitz  # pip install pymupdf

# 1) PDF to images
pdf_path = "./data/sample.pdf"
doc = fitz.open(pdf_path)
imgs = []
for i in range(min(3, len(doc))):  # first 3 pages
    pix = doc[i].get_pixmap(dpi=200)
    p = f"./data/pages/page_{i}.png"
    pix.save(p)
    imgs.append(Image.open(p))

# 2) OCR with TrOCR (printed)
ocr = pipeline("image-to-text", model="microsoft/trocr-base-printed")
pages_text = "\n\n".join(ocr(img)[0]["generated_text"] for img in imgs)

# 3) Summarize with a small model
summ = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
summary = summ(pages_text[:3000], max_length=150, min_length=60, do_sample=False)[0]["summary_text"]

# 4) Bullet keypoints via T5 prompt
gen = pipeline("text2text-generation", model="google/flan-t5-small")
keypoints = gen("Make 5 concise bullet points:\n" + summary, max_length=128)[0]["generated_text"]

print("SUMMARY:\n", summary)
print("\nKEYPOINTS:\n", keypoints)
