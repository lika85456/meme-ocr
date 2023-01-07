# Meme OCR
Research of OCR engines and filters for memes. Uses Tesseract and EasyOCR to build a REST API server, which can be used to extract text from memes.

# How to the server
Before running the server change API_KEY in Dockerfile.
```
docker build . -t <image_name>
docker run -it -p 5000:5000 <image_name>
```

# Usage
Use POST requests to `/tesseract` or `/easyocr` with a `file` form-data parameter to get the text from an image. Also use `Authorization` header with `<token>` to authenticate the request.
You can set the token in the Dockerfile.

**Example request:**
```
curl --location --request POST 'http://0.0.0.0:5000/tesseract' \
--header 'Authorization: <token>' \
--form 'file=@"<path_to_file>"'
```

**Example response:**
```

Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do ei-
usmod tempor incididunt ut labore et dolore magna aliqua. Ut
enim ad minim veniam, quis nostrud exercitation ullamco laboris

nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla
pariatur. Excepteur sint occaecat cupidatat non proident, sunt in
culpa qui officia deserunt mollit anim id est laborum. Lorem ipsum
dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor
incididunt ut labore et dolore magna aliqua. Ut enim ad minim
```