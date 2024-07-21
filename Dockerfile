FROM python:3.11-slim
# FROM --platform=linux/amd64 python:3.11-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

EXPOSE 8080

#This is if you using uvicorn with fast API
#CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]

#this is if you using gunicorn with Flask
CMD exec gunicorn --bind :8080 --workers 1 --threads 8 --timeout 0 app.main:app

#this is if you want to deploy it with cloud run
#CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app.main:app