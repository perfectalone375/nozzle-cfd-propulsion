FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install C++17 compiler for native high-speed MacCormack CFD kernel
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
    && g++ -O3 -std=c++17 -Wall -Wextra -o /usr/local/bin/maccormack_cfd cpp/maccormack_cfd.cpp

ENTRYPOINT ["nozzle-cfd"]
CMD ["--propellant", "METHALOX", "--chamber-pressure-mpa", "10.0", "--ambient-pressure-kpa", "101.325"]
