run:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up

up:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

down:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml down

build:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml build $(SERVICE)

logs:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml logs -f $(SERVICE)

run-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.ics-iqbert up

up-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.ics-iqbert up -d

down-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.ics-iqbert down

logs-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.ics-iqbert logs

build-prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml build $(SERVICE)

push-iqb-registry:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml --env-file .env.ics-iqbert build
	docker login scm.cms.hu-berlin.de:4567
	bash -c 'source .env.ics-iqbert && docker push $${REGISTRY_PATH}ics-iqbert-backend:$${TAG}'
	bash -c 'source .env.ics-iqbert && docker push $${REGISTRY_PATH}ics-iqbert-worker:$${TAG}'
	docker logout
