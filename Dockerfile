FROM python:3.10-slim

ENV PYTHONUNBUFFERED True

ENV APP_HOME /app

ENV PORT 8080

WORKDIR $APP_HOME

COPY . ./

RUN pip install -r requirements.txt

RUN sudo apt-get install libglu1 -y

RUN sudo apt-get update && sudo apt-get install ffmpeg libsm6 libxext6  -y

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app