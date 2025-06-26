FROM python:3.11

RUN mkdir /data
RUN mkdir /results

COPY ics /app/ics
COPY lib /app/lib

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

CMD ["rq", "worker"]
