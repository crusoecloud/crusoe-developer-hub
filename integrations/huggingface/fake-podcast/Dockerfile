# ============================================================
# FakePod — Hugging Face Spaces Docker image
# Multi-stage: build React frontend, then serve with FastAPI
# ============================================================

# --- Stage 1: Build React frontend ---
FROM node:20-slim AS frontend-build

WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# --- Stage 2: Python runtime ---
FROM python:3.11-slim

# HF Spaces runs as user "user" with uid 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Install Python dependencies
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./

# Copy built frontend into backend/static
COPY --from=frontend-build /app/frontend/dist ./static

# HF Spaces expects port 7860
ENV PORT=7860
EXPOSE 7860

# Switch to non-root user
USER user

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
