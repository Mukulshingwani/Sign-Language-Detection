FROM python:3.9.16-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --trusted-host pypi.python.org -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


EXPOSE 80

ENV NAME project

CMD ["python", "main.py"]
