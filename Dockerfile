# The frozen third-version image must already be loaded locally under this tag.
FROM zhixun-engine:workspace-qwen-amd64-v1
USER root
RUN rm -rf /opt/zhixun && mkdir -p /app /workspace && chown 10001:10001 /workspace && ln -s python /opt/venv/bin/python3
COPY --chown=0:0 . /app/
ENV PYTHONPATH=/app:/app/src:/app/runtime PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
USER 10001:10001
ENTRYPOINT ["/opt/venv/bin/python", "/app/harness.py"]
CMD ["describe"]
