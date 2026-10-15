# Docker Directory

This directory contains Docker-related files for containerized deployment of ALchemist.

## Files

**`Dockerfile`**
- Docker image definition based on Python 3.11-slim
- Includes Node.js for building React frontend during image build
- Automatically builds frontend via `build_tools/build_hooks.py`
- Installs ALchemist from wheel with bundled UI

**`docker-compose.yml`**
- Docker Compose configuration for easy deployment
- Includes volume mounts for logs and cache persistence
- Configurable CORS origins for production domains

**`.dockerignore`**
- Excludes unnecessary files from Docker build context
- Optimizes build speed and image size

## Usage

### Quick Start

From the project root:

```bash
cd docker
docker-compose up --build
```

Access at: http://localhost:8000

### Detailed Steps

**Build the image:**
```bash
cd docker
docker-compose build
```

**Start the container:**
```bash
docker-compose up
```

**Run in background (detached mode):**
```bash
docker-compose up -d
```

**View logs:**
```bash
docker-compose logs -f
```

**Stop the container:**
```bash
docker-compose down
```

## How It Works

The Dockerfile:
1. Installs Node.js and npm (needed for frontend build)
2. Copies the entire project into the container
3. Runs `python -m build` which triggers `build_tools/build_hooks.py`
4. `build_tools/build_hooks.py` automatically runs `npm ci && npm run build`
5. Frontend is bundled into `api/static/` and included in the wheel
6. Installs the wheel with pre-built UI
7. Cleans up build artifacts to reduce image size

**Result:** Users get a working web UI without Node.js in the final runtime!

## Configuration

### Production Deployment

Edit `docker-compose.yml`:

**Update CORS origins for your domain:**
```yaml
environment:
  - ALLOWED_ORIGINS=http://alchemist.example.org,https://alchemist.example.org
```

**Add resource limits (optional):**
```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

**Use environment file:**
```bash
# Create .env file
echo "ALLOWED_ORIGINS=http://alchemist.example.org" > .env

# Update docker-compose.yml
services:
  alchemist-api:
    env_file: .env
```

### Reverse Proxy Setup

For production with SSL/HTTPS, use nginx or Caddy:

**Example nginx config:**
```nginx
server {
    listen 443 ssl;
    server_name alchemist.example.org;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Health Check

The container includes a health check at `/health`. Docker will automatically monitor this endpoint every 30 seconds.

Check health status:
```bash
docker ps  # Look for "healthy" status
```

## Troubleshooting

**Frontend not loading:**
```bash
# Check if static files exist in the image
docker run --rm alchemist:0.3.0 ls -la /app/api/static/

# Should see index.html and assets/
```

**Build fails:**
```bash
# Build with verbose output
docker-compose build --progress=plain --no-cache
```

**Container exits immediately:**
```bash
# Check logs
docker-compose logs

# Run interactively for debugging
docker run -it --rm alchemist:0.3.0 /bin/bash
```

## Image Size Optimization

The Dockerfile includes several optimizations:
- Uses `python:3.11-slim` base image
- `--no-cache-dir` for pip installs
- Removes Node.js build artifacts after build
- Cleans up `/var/lib/apt/lists/`

Current image size: ~1.2-1.5GB (depending on dependencies)

## Alternative: Use PyPI Wheel

For even simpler deployment without building:

```dockerfile
FROM python:3.11-slim

# Install from PyPI (includes pre-built UI)
RUN pip install alchemist-nrel

CMD ["python", "-m", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

This is faster but doesn't include the latest development changes.
