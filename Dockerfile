# syntax=docker/dockerfile:1
# check=skip=FromPlatformFlagConstDisallowed
# Multi-stage Dockerfile for sweets (CPU build).
# Builds for linux/amd64 only (arm64 not supported due to isce3 dependency).
#
# Build:
#   docker build -t sweets:latest .
#
# Run:
#   docker run --rm -v $PWD:/work sweets:latest sweets run sweets_config.yaml

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies using pixi
# ---------------------------------------------------------------------------
# --platform=linux/amd64 is pinned because sweets' pixi workspace supports
# only linux-64 / osx-arm64, and isce3 has no linux-aarch64 conda build.
# On Apple Silicon Docker Desktop this means the container runs emulated.
FROM --platform=linux/amd64 ghcr.io/prefix-dev/pixi:0.65.0 AS install

# git + ca-certificates needed for pip to install git+ pypi dependencies
# (scottstanie/s1-reader, scottstanie/COMPASS, scottstanie/opera-utils,
# scottstanie/dolphin, scottstanie/spurt).
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Set version for setuptools-scm (since .git is excluded from build context)
ARG VERSION=0.0.0.dev0
# Package-scoped form so we only pin sweets' own version; the unscoped
# SETUPTOOLS_SCM_PRETEND_VERSION would leak into every setuptools_scm build
# in this layer (including uv's git builds of dolphin / opera-utils forks),
# stamping them all as 0.0.0.dev0 and breaking version constraints.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SWEETS=${VERSION}

# Copy only dependency files first (for layer caching)
COPY pyproject.toml pixi.lock LICENSE README.md ./

# Create minimal source structure so setuptools metadata resolves
RUN mkdir -p src/sweets && touch src/sweets/__init__.py

# Install dependencies with cache mount for rattler
RUN --mount=type=cache,target=/root/.cache/rattler/cache,sharing=private \
    pixi install -e default

# ---------------------------------------------------------------------------
# Stage 2: Build the package
# ---------------------------------------------------------------------------
FROM install AS build

# Re-declare ARG (needed in each stage that uses it)
ARG VERSION=0.0.0.dev0
# Package-scoped form so we only pin sweets' own version; the unscoped
# SETUPTOOLS_SCM_PRETEND_VERSION would leak into every setuptools_scm build
# in this layer (including uv's git builds of dolphin / opera-utils forks),
# stamping them all as 0.0.0.dev0 and breaking version constraints.
ENV SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SWEETS=${VERSION}

# Copy the full source code
COPY . .

# Reinstall with real source to get the actual package installed
RUN --mount=type=cache,target=/root/.cache/rattler/cache,sharing=private \
    pixi install -e default

# Build isce3 from source to work around conda-forge linux-64 resamp() SIGSEGV.
# Opt-in via --build-arg BUILD_ISCE3_FROM_SOURCE=true.
ARG ISCE3_REPO=https://github.com/scottstanie/isce3.git
ARG ISCE3_BRANCH=scott-develop
ARG BUILD_ISCE3_FROM_SOURCE=false
RUN if [ "$BUILD_ISCE3_FROM_SOURCE" = "true" ]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        g++ cmake ninja-build pkg-config && \
      rm -rf /var/lib/apt/lists/* && \
      git clone --branch ${ISCE3_BRANCH} --depth 1 ${ISCE3_REPO} /isce3 && \
      cmake -B /isce3/build -S /isce3 -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/app/.pixi/envs/default \
        -DCMAKE_PREFIX_PATH=/app/.pixi/envs/default \
        -DCMAKE_CXX_FLAGS="-include cstdint" \
        -DWITH_CUDA=NO && \
      cmake --build /isce3/build --parallel $(nproc) && \
      cmake --build /isce3/build --target install && \
      rm -rf /isce3; \
    fi

# Create activation script using pixi shell-hook
RUN pixi shell-hook -e default > /activate.sh && \
    chmod +x /activate.sh

# ---------------------------------------------------------------------------
# Stage 3: Minimal production image
# ---------------------------------------------------------------------------
FROM --platform=linux/amd64 ubuntu:24.04 AS production

LABEL org.opencontainers.image.description="Container for sweets InSAR workflows"
LABEL org.opencontainers.image.authors="Scott Staniewicz <scott.j.staniewicz@jpl.nasa.gov>"
LABEL org.opencontainers.image.url="https://github.com/isce-framework/sweets"
LABEL org.opencontainers.image.source="https://github.com/isce-framework/sweets"
LABEL org.opencontainers.image.licenses="BSD-3-Clause OR Apache-2.0"

# Minimal runtime dependencies (ca-certificates for https downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the production environment and source from build stage
COPY --from=build /app/.pixi/envs/default /app/.pixi/envs/default
COPY --from=build /app/src /app/src
COPY --from=build /activate.sh /activate.sh

WORKDIR /work
ENV HOME=/work

# Entrypoint activates the pixi environment so `sweets` is on PATH
RUN printf '#!/bin/bash\nset -e\nsource /activate.sh\nexec "$@"\n' > /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]

# Default command runs `sweets --help`; override with your own command, e.g.
#   docker run --rm -v $PWD:/work sweets:latest sweets run sweets_config.yaml
CMD ["sweets", "--help"]
