#!/bin/bash

docker compose exec app uv run python src/main.py --model-name gpt-4.1 --prompting-strategy basic --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o --prompting-strategy basic --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o-mini --prompting-strategy basic --sample-size 50 --use-seed False

docker compose exec app uv run python src/main.py --model-name gpt-4.1 --prompting-strategy few_shot --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o --prompting-strategy few_shot --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o-mini --prompting-strategy few_shot --sample-size 50 --use-seed False

docker compose exec app uv run python src/main.py --model-name gpt-4.1 --prompting-strategy chain_of_thought --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o --prompting-strategy chain_of_thought --sample-size 50 --use-seed False
docker compose exec app uv run python src/main.py --model-name gpt-4o-mini --prompting-strategy chain_of_thought --sample-size 50 --use-seed False

docker compose exec app uv run python src/main.py --model-name o4-mini --prompting-strategy chain_of_thought --sample-size 20 --use-seed False
docker compose exec app uv run python src/main.py --model-name o4-mini --prompting-strategy chain_of_thought --sample-size 20 --use-seed False
docker compose exec app uv run python src/main.py --model-name o4-mini --prompting-strategy chain_of_thought --sample-size 20 --use-seed False
