# Hugging Face Spaces (Docker SDK) builds and runs this directly.
#
# One thing worth calling out: LightGBM needs the OpenMP runtime at import
# time. On macOS that's libomp (we hit this exact issue running locally -
# "Library not loaded: @rpath/libomp.dylib" - and fixed it with
# `brew install libomp`). On a Debian-based container, the equivalent
# library is called libgomp1, and it isn't installed by default in the
# slim base image. Skipping this line reproduces the same class of crash
# inside the container that we saw locally, just with a different library
# name - so it's installed here, before anything else, as root.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces containers run as a non-root user by convention (UID 1000) -
# running as root works but HF's own guidance is to avoid it, and it also
# matches how most production container setups are configured anyway.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app

# HF Spaces expects the app to listen on port 7860 by default (configurable,
# but 7860 is the convention and needs no extra config on their end)
EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
