FROM python:3.13-slim AS production

ARG UID=1000
ARG GID=1000

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

RUN mkdir -p /data /code/outputs
WORKDIR /code

ENV PATH="/code/.venv/bin:$PATH" \
    PYTHONPATH="/code"

RUN groupadd --non-unique -g $GID nonroot && \
    useradd --non-unique --no-log-init -u $UID -g $GID -m nonroot && \
    chown -R nonroot:nonroot /data /code

USER nonroot

RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=$UID,gid=$GID \
    --mount=type=bind,source=pyproject.toml,target=/code/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/code/uv.lock \
    uv sync --locked --no-install-project --no-dev

COPY --chown=nonroot:nonroot ./app /code/app
COPY --chown=nonroot:nonroot ./tests /code/tests

ENV PROMPT_COLOUR=31
RUN echo 'PS1="\033[${PROMPT_COLOUR}m[convfinqa-$(echo $CONFIG_ENV)] \W \\$ \033[0m"' >> /home/nonroot/.bashrc

FROM production AS development
USER nonroot

ENV CONFIG_ENV=dev \
    PROMPT_COLOUR=32

RUN --mount=type=cache,target=/home/nonroot/.cache/uv,uid=$UID,gid=$GID \
    --mount=type=bind,source=pyproject.toml,target=/code/pyproject.toml \
    --mount=type=bind,source=uv.lock,target=/code/uv.lock \
    uv sync --locked --dev

RUN echo 'alias format="ruff format app/ tests/ && ruff check --fix app/ tests/"' >> /home/nonroot/.bashrc && \
    echo 'alias code-checks="ruff format --check app/ tests/ && ruff check app/ tests/ && pyrefly check"' >> /home/nonroot/.bashrc && \
    echo 'alias run-tests="coverage run -m pytest && coverage report"' >> /home/nonroot/.bashrc && \
    echo 'alias pipeline="code-checks && run-tests"' >> /home/nonroot/.bashrc && \
    echo 'alias all-checks="format && code-checks && run-tests"' >> /home/nonroot/.bashrc
