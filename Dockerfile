# syntax=docker/dockerfile:1.2
FROM python:3.10
# put you docker configuration here
# syntax=docker/dockerfile:1.2
RUN apt-get -y update
COPY . /app
WORKDIR /app
RUN pip install --no-cache-dir -r requirements.txt
ENV PORT 8080
EXPOSE 8080
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]