# ============================
# Stage 1: Build dependencies
# ============================
# A "multi-stage" build: this first stage installs compilers and dev headers
# (build-essential, gcc) needed only to COMPILE some Python packages' native
# extensions. None of that toolchain is needed to RUN the app afterwards, so
# it is discarded once this stage finishes -- only the installed packages
# and application code are carried into Stage 2 below. This keeps the final
# image significantly smaller than installing everything in one stage.
FROM python:3.12.10-slim AS builder

WORKDIR /build

# Install build tools only here
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements/ ./requirements/

# Install into a temporary location (--prefix=/install) rather than the
# system site-packages, so Stage 2 can cherry-pick just this folder without
# also copying pip's build caches or the compilers installed above.
#
# NOTE: requirements/prod.txt and requirements/constraints.txt should be kept
# in sync with the versions pinned in requirements/base.txt (scikit-learn,
# joblib, mlxtend, etc.) -- a mismatch here reintroduces the same
# InconsistentVersionWarning / mlxtend bug that pinning base.txt was meant to
# prevent, just inside this container instead of inside a local venv.
RUN pip install --no-cache-dir --prefix=/install -r requirements/base.txt -c requirements/constraints.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements/prod.txt -c requirements/constraints.txt

COPY api.py .
# Baking the .joblib model files into the image means they are versioned
# together with the code that serves them -- but it also means this image
# must be REBUILT (`docker compose build flask-gunicorn-app`) any time the
# models under model/ are re-fit, or the container will keep serving the
# old ones.
COPY model/ model/

# ============================
# Stage 2: Runtime image
# ============================
FROM python:3.12.10-slim

WORKDIR /opt/app-smm
ENV PYTHONPATH=/opt/app-smm

# Copy only the built Python packages from the builder stage (no compilers,
# no apt cache, no pip cache -- those never leave Stage 1).
COPY --from=builder /install /usr/local

# Copy application code (and the model/ folder copied there in Stage 1)
COPY --from=builder /build /opt/app-smm

EXPOSE 8000

# 5 Gunicorn worker processes -- a fixed, reasonable default for a small
# lab container. In a real deployment this is normally sized from the
# machine's CPU count, e.g. (2 x number_of_cores) + 1.
CMD ["gunicorn", "--chdir", "/opt/app-smm", "-w", "5", "-b", "0.0.0.0:8000", "api:app"]
