FROM python:3.8

WORKDIR /app/

COPY pyproject.toml /app/

ENV PYTHONPATH=${PYTHONPATH}:${PWD}

RUN pip3 install poetry
RUN poetry config virtualenvs.create false

RUN poetry install --no-root
