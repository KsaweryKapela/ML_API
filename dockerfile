FROM python:3.10.11-bullseye

WORKDIR /api
COPY . /api

RUN pip install --upgrade pip
RUN pip install --default-timeout=100 future
RUN pip install -r requirements.txt

EXPOSE 3000
CMD ["python3", "api.py"]