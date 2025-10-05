# 1. Base Image: Start with a lightweight Python 3.9 image
FROM python:3.9-slim

# 2. Install System Dependencies (git & git-lfs)
RUN apt-get update && \
    apt-get install -y --no-install-recommends git curl && \
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash && \
    apt-get install -y git-lfs && \
    rm -rf /var/lib/apt/lists/*

# 3. Clone the Repository from GitHub
# This command brings the entire repository, including the .git folder, into the /app directory.
RUN git clone https://github.com/sundeep3257/Monteggia_Dx.git /app

# 4. Set the Working Directory
WORKDIR /app

# 5. Download LFS Files
# Now that we are inside a Git repository, this command will succeed.
RUN git lfs pull

# 6. Install Python Requirements
RUN pip install --no-cache-dir -r requirements.txt

# 7. Set the Start Command
# Use Gunicorn to run the Flask app. OnRender sets the $PORT variable.
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:$PORT", "application:application"]
