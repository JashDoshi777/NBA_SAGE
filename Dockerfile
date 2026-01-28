# syntax=docker/dockerfile:1

# ============================================================================
# NBA Sage Predictor - Hugging Face Spaces Docker Image
# ============================================================================

FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the web frontend package files
COPY web/package*.json ./web/
WORKDIR /app/web

# Install Node dependencies (use npm install for robustness)
RUN npm install

# Copy web source files and build
COPY web/ .
RUN npm run build

# Move back to app root
WORKDIR /app

# Copy built frontend to static folder
RUN mkdir -p static && cp -r web/dist/* static/

# Copy Python source code
COPY src/ ./src/
COPY api/ ./api/
COPY server.py .

# Copy data and models
COPY data/processed/ ./data/processed/
COPY data/api_data/ ./data/api_data/
COPY models/ ./models/

# Create data directories for runtime
RUN mkdir -p data/predictions data/injuries

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/api/health || exit 1

# Run the production server
CMD ["python", "server.py"]
