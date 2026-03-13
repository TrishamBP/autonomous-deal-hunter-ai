FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install uv && uv pip install -e .

ENV PYTHONUNBUFFERED=1

EXPOSE 7860

CMD ["python", "-m", "core.main"]
