# ==============================
# BASE IMAGE
# ==============================
FROM python:3.10-slim

# ==============================
# ENV SETTINGS (IMPORTANT)
# ==============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ==============================
# WORKDIR
# ==============================
WORKDIR /app

# ==============================
# SYSTEM DEPENDENCIES
# ==============================
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# ==============================
# INSTALL PYTHON DEPENDENCIES
# ==============================
COPY requirements.txt .

RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ==============================
# COPY PROJECT
# ==============================
COPY . .

# ==============================
# EXPOSE PORT
# ==============================
EXPOSE 8000

# ==============================
# RUN SERVER
# ==============================
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
