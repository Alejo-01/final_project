FROM python:3.9

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt

CMD ["uvicorn", "Parte_B.main:app", "--host=0.0.0.0", "--port=7860"]