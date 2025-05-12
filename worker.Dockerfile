FROM python:3.11

COPY data /app/data

COPY ics /app/ics
COPY src /app/src

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install transformers datasets torch transformers[torch]

#CMD ["rq", "worker"]
