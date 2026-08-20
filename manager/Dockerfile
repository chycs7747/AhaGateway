FROM python:3.12-slim

WORKDIR /app

# 의존성 먼저 — 코드만 바뀌면 이 레이어는 캐시에서 재사용된다
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY manager/ manager/

EXPOSE 9100
CMD ["fastapi", "run", "manager/main.py", "--port", "9100"]