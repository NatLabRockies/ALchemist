# ALchemist Docker Image for FastAPI Server
# This builds a container that runs the ALchemist REST API with pre-built web UI

# Start from official Python 3.11 image
FROM python:3.11-slim

# Set metadata labels
LABEL maintainer="caleb.coatney@nrel.gov"
LABEL description="ALchemist Active Learning Toolkit - REST API Server"
LABEL version="0.3.3"

# Set working directory inside the container
WORKDIR /app

# Install system dependencies
# - gcc, g++, make: Required for compiling some Python packages (numpy, scipy, torch)
# - nodejs, npm: Required to build React frontend during installation
# - git: Needed if any dependencies come from git repos
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Copy the entire project
COPY . .

# Build the React frontend first
WORKDIR /app/alchemist-web
RUN npm install && \
    npm run build

# Copy built frontend to API static directory
WORKDIR /app
RUN mkdir -p api/static && \
    cp -r alchemist-web/dist/* api/static/

# Build and install ALchemist package
RUN pip install --no-cache-dir --upgrade pip build && \
    python -m build && \
    pip install --no-cache-dir dist/*.whl

# Verify the frontend was built
RUN test -f api/static/index.html || (echo "ERROR: Frontend not built!" && exit 1)

# Clean up build artifacts to reduce image size
RUN rm -rf /app/dist /app/build /app/alchemist-web/node_modules /app/alchemist-web/dist

# Create directories for logs and cache
RUN mkdir -p /app/logs /app/cache

# Expose port 8000 (FastAPI default)
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Health check - Docker will periodically check if the container is healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the FastAPI server when container starts
# Using alchemist-web entry point (installed from wheel)
CMD ["alchemist-web", "--production"]
