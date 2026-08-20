FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ gateway/

EXPOSE 9000
CMD ["fastapi", "run", "gateway/main.py", "--port", "9000"]