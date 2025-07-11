# Target environment for Linux.
FROM rockylinux:9 AS target-linux
WORKDIR /repo
ARG APP_VERSION
RUN echo $APP_VERSION > .version

# Install required packages.
COPY --chmod=0755 scripts/target.sh .
RUN ./target.sh && rm target.sh

# Download continuous integration dependencies.
FROM target-linux AS ci
COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=source=uniopt,target=/repo/uniopt,rw \
    python -m venv .venv \
    && source .venv/bin/activate \
    && python -m pip install -e .[ci]

# Type checking.
FROM ci AS type-checker
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN --mount=source=uniopt,target=/repo/uniopt \
    mkdir -p /out/report \
    && source .venv/bin/activate \
    && python -m ty check --exit-zero uniopt > /out/report/type_check.txt

# Type checking collection.
FROM scratch AS type
COPY --from=type-checker /out/* /
