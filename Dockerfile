FROM python:3.10

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./app /code/app

COPY ./weights /code/weights

COPY ./sample-footage /code/sample-footage

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]