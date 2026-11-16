# The inkling gateway: Anthropic/OpenAI API in front of the TensorFold inkling2 server.
# Stdlib-only Python; no dependencies to install.
FROM python:3.12-slim
WORKDIR /gateway
COPY inkling_gateway.py resume_spool.py stream-resume.enabled ./
ENTRYPOINT ["python3", "/gateway/inkling_gateway.py"]
