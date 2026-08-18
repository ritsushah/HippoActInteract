.PHONY: build test fetch embed arch train viz atlas benchmark shell clean

build:
	docker compose build

test:
	docker compose run --rm pipeline pytest tests -v --log-cli-level=INFO

fetch:
	docker compose run --rm --entrypoint python pipeline -m src.data_fetcher

embed:
	docker compose run --rm --entrypoint python pipeline -m src.embedder

arch:
	docker compose run --rm --entrypoint python pipeline -m src.gnn_model

train:
	docker compose run --rm --entrypoint python pipeline -m src.train

viz:
	docker compose run --rm --entrypoint python pipeline -m src.visualize

atlas:
	docker compose run --rm --entrypoint python pipeline -m src.evidence_atlas

benchmark:
	docker compose run --rm --entrypoint python pipeline -m src.physical_benchmark

shell:
	docker compose run --rm --entrypoint bash pipeline

clean:
	docker compose down --rmi all -v --remove-orphans
