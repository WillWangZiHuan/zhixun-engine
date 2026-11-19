FROM zhixun-engine:sdk-1.0.2-amd64-v1
COPY --chown=0:0 . /app/
LABEL org.opencontainers.image.version="sdk-1.0.4" zhixun.sdk-source.sha256="7afd282d6b5fd889d64d0515a67d169806970e36200689b18d2040026167a4af"
