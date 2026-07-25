# Hugging Face deprecated the native "sdk: streamlit" Space type in favor
# of Docker SDK + a Streamlit template - this Dockerfile follows their
# current recommended pattern rather than the older built-in option, so it
# won't break if the native SDK path gets removed entirely later.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# Streamlit's own default port, and the convention HF's Streamlit-on-Docker
# template uses - keeping it avoids any port-mismatch surprises
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
