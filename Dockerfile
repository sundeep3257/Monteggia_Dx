# 1. Base Image: Start with a lightweight Python 3.9 image
FROM python:3.9-slim

# 2. Install System Dependencies (including git-lfs)
# This section creates a writable environment to install what we need.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/*

# 3. Set Up Application Directory
WORKDIR /app

# 4. Copy and Install Python Requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy Application Code
COPY . .

# 6. Download LFS Files
# This runs git lfs pull inside the container to get the real model files.
RUN git lfs pull

# 7. Set the Start Command
# OnRender will automatically set the $PORT variable.
# We use Gunicorn, a production-ready server for Flask.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:$PORT", "application:application"]
