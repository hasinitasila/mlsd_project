# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── Set working directory ────────────────────────────────────────────────────
WORKDIR /app

# ── Install dependencies ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt

# ── Copy source code ─────────────────────────────────────────────────────────
COPY train.py .
COPY app.py   .

# ── Train model at build time so artifacts are baked in ──────────────────────
# (Alternatively, mount model/ as a volume in production)
COPY model/ ./model/

# ── Expose Streamlit default port ────────────────────────────────────────────
EXPOSE 8501

# ── Healthcheck ──────────────────────────────────────────────────────────────
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ── Run the app ──────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true"]
