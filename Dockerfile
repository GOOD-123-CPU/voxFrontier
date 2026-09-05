# VoxFrontier container image.
#
# Build:  docker build -t voxfrontier .
# Run:    docker run --rm -it -v "$PWD/output:/app/output" voxfrontier vxf run-all
#
# The image contains only open-source components (Python + scientific stack);
# no proprietary data or platform assets are copied into it.

FROM python:3.11-slim

LABEL org.opencontainers.image.title="VoxFrontier"
LABEL org.opencontainers.image.description="Causal and efficiency frontier analytics for voice-driven livestream performance (DEA, Shapley, DML, conformal inference) on privacy-safe synthetic data"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/GOOD-123-CPU/voxFrontier"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install system deps needed by the scientific stack (none beyond stdlib
# wheels are required on py3.11-slim for numpy/pandas/scipy/scikit-learn).
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --no-cache-dir .

# Ship the default config; users can override by mounting their own.
COPY config ./config
COPY tests ./tests

# Run as a non-root user for safer defaults.
RUN useradd --create-home --shell /bin/bash vxuser \
    && chown -R vxuser:vxuser /app
USER vxuser

CMD ["vxf", "run-all"]
