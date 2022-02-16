FROM python:3.9-slim

EXPOSE 8888

COPY ./app /app
COPY requirements.txt /app/
WORKDIR /app

RUN pip install -r requirements.txt && rm requirements.txt

CMD jupyter notebook --port=8888 --ip=0.0.0.0 --NotebookApp.token='' --allow-root