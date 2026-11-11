FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install C++20 toolchain for native MacCormack, HLLC+ FVM, and shared library kernels
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY cpp ./cpp
COPY matlab ./matlab
COPY tests ./tests

RUN pip install --no-cache-dir -e ".[dev]" \
    && make -C cpp all \
    && cp cpp/maccormack_cfd cpp/hllc_fvm_cfd cpp/axisymmetric_ns2d /usr/local/bin/

ENTRYPOINT ["nozzle-cfd"]
CMD ["--propellant", "METHALOX", "--chamber-pressure-mpa", "10.0", "--ambient-pressure-kpa", "101.325"]
