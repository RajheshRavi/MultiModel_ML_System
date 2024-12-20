FROM python:latest

WORKDIR /app/multi_model/

COPY app.py ./src/
COPY resources/resources.txt ./src/resources/
COPY modelSelector.py ./src/
COPY preprocess.py ./src/
COPY errorCalc.py ./src/


RUN pip install -r ./src/resources/resources.txt

CMD ["python", "./src/app.py"]