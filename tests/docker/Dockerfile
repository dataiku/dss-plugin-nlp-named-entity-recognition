ARG PYTHON_VERSION=3.12
FROM python:${PYTHON_VERSION}-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /plugin

# Copy requirements files first (for better layer caching)
COPY code-env/python/spec/requirements.txt /plugin/requirements-plugin.txt
COPY tests/python/unit/requirements.txt /plugin/requirements-test.txt

# USE_CACHE=true enables BuildKit pip cache for faster local iteration
# USE_CACHE=false (default) uses --no-cache-dir for clean CI builds
ARG PYTHON_VERSION
ARG USE_CACHE=false

# Install dependencies without cache (default, CI)
RUN if [ "$USE_CACHE" != "true" ]; then \
      pip install --upgrade pip && \
      pip install --no-cache-dir -r requirements-test.txt && \
      pip install --no-cache-dir -r requirements-plugin.txt; \
    fi

# Install dependencies with BuildKit cache mount (local dev)
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-${PYTHON_VERSION} \
    if [ "$USE_CACHE" = "true" ]; then \
      pip install --upgrade pip && \
      pip install -r requirements-test.txt && \
      pip install -r requirements-plugin.txt; \
    fi

# Copy source code and resources
COPY python-lib/ /plugin/python-lib/
COPY resource/ /plugin/resource/
COPY tests/python/unit/ /plugin/tests/python/unit/

# Set environment variables
ENV PYTHONPATH=/plugin/python-lib

# Run tests with system info
CMD echo "=== System Info ===" && \
    echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'=' -f2)" && \
    echo "Architecture: $(uname -m)" && \
    echo "Python: $(python --version)" && \
    echo "Pip: $(pip --version)" && \
    echo "==================" && \
    pytest tests/python/unit -v