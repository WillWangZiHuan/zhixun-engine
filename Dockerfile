FROM python:3.12.16-slim
RUN python -m venv /opt/venv && /opt/venv/bin/pip install --no-cache-dir numpy==1.26.4
ENV PATH=/opt/venv/bin:$PATH PYTHONPATH=/app:/app/src:/app/runtime PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HOME=/work
RUN mkdir -p /workspace /work && chown 10001:10001 /workspace /work
COPY . /app/
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT ["/opt/venv/bin/python","/app/harness.py"]
CMD ["describe"]
