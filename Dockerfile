FROM python:3.12
WORKDIR /home
COPY ./requirements.txt /home
COPY ./app /home/app
RUN pip install --no-cache-dir --upgrade -r /home/requirements.txt

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
