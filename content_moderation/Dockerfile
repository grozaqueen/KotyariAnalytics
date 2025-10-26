FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app/content_moderation

ENV PYTHONUNBUFFERED=1
EXPOSE 8088
CMD ["python", "-m", "content_moderation.server"]
