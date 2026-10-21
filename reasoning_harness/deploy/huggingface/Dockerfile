# Hugging Face Space (Docker SDK) for the Puzzle Arena. CPU only.
FROM python:3.12-slim

RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir "numpy>=1.26,<3" "chess>=1.10,<2" "opencv-python-headless>=4.9,<5" "pillow>=10,<12"

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user . .

ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["python", "-m", "arena.cli", "ui", "--host", "0.0.0.0", "--port", "7860", \
     "--artifacts", "artifacts/arena", "--tests", "benchmarks/arena"]
