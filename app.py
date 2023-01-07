import os
from flask import Flask, request
from src import reader

app = Flask(__name__)

# get key from environment variable
apiKey = (
    os.environ.get("API_KEY")
    or "768a2361a3058210bc07204639f79cb72aa98f0f7ff648ba101a38ccb543d6f76c548c1f39549e9afe5752cb77b93a580fb7d24784dc08cf1c67a71130eeb081"
)


def isAuthorized(req):
    # check request authorization code
    key = req.headers.get("Authorization")
    if key == apiKey:
        return True
    else:
        return False


@app.post("/easyocr")
def easyocr():
    # check authorization
    if not isAuthorized(request):
        return "Unauthorized", 401

    image_id = os.urandom(16).hex()
    temp_file_name = f"/tmp/{image_id}.png"

    file = request.files["file"]
    file.save(temp_file_name)

    return "\n".join(reader.EasyOCRReader().read(temp_file_name))


@app.post("/tesseract")
def tesseract():
    # check authorization
    if not isAuthorized(request):
        return "Unauthorized", 401

    image_id = os.urandom(16).hex()
    temp_file_name = f"/tmp/{image_id}.png"

    file = request.files["file"]
    file.save(temp_file_name)

    return "\n".join(reader.TesseractReader().read(temp_file_name))
