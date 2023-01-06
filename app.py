from flask import Flask, request
from src.reader import EasyOCRReader, TesseractReader

app = Flask(__name__)


@app.post("/easyocr")
def easyocr():
    f = request.files["file"]
    f.save("/tmp/test.png")

    return "\n".join(EasyOCRReader().read("/tmp/test.png"))


@app.post("/tesseract")
def tesseract():
    f = request.files["file"]
    f.save("/tmp/test.png")

    return "\n".join(TesseractReader().read("/tmp/test.png"))
