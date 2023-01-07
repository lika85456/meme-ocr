FROM python:3.10
COPY . /app
WORKDIR /app


RUN pip install --upgrade pip
RUN pip install pipenv

ENV API_KEY=768a2361a3058210bc07204639f79cb72aa98f0f7ff648ba101a38ccb543d6f76c548c1f39549e9afe5752cb77b93a580fb7d24784dc08cf1c67a71130eeb081
ENV PROJECT_DIR /app
ENV CUDA=None

RUN apt-get update \
  && apt-get -y install tesseract-ocr

RUN pipenv install --system --deploy

# RUN pip3 install flask --no-cache-dir
# RUN pip3 install easyocr --no-cache-dir
# RUN pip3 install pytesseract --no-cache-dir
# RUN pip3 install opencv-python-headless==4.6.0.66 --no-cache-dir
# RUN pip3 install numpy --no-cache-dir
# RUN pip3 install gunicorn --no-cache-dir

# download EasyOCR models
RUN wget https://github.com/JaidedAI/EasyOCR/releases/download/v1.3/english_g2.zip
RUN wget https://github.com/JaidedAI/EasyOCR/releases/download/pre-v1.1.6/craft_mlt_25k.zip
RUN mkdir ~/.EasyOCR
RUN mkdir ~/.EasyOCR/model
RUN unzip english_g2.zip -d ~/.EasyOCR/model
RUN unzip craft_mlt_25k.zip -d ~/.EasyOCR/model

EXPOSE 5000

ENTRYPOINT FLASK_APP=/app/app.py flask run --host=0.0.0.0