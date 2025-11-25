# Multi-stage build for LLM Council

# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend
FROM python:3.10-slim
WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy Python project files
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Copy backend code
COPY backend/ ./backend/
COPY main.py ./

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

# Expose port
EXPOSE 8001

# Run the application
CMD [".venv/bin/python", "-m", "backend.main"]
