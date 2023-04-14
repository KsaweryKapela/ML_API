FROM python:3.10.11-bullseye

WORKDIR /app
COPY . /app

RUN pip install --upgrade cython
RUN pip install --upgrade pip
RUN pip install --default-timeout=100 future
RUN pip install -r requirements.txt

EXPOSE 3000
CMD python ./api/API.py