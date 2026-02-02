# Makefile variables set automatically
plugin_id=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['id']).replace('/',''))"`
plugin_version=`cat plugin.json | python -c "import sys, json; print(str(json.load(sys.stdin)['version']).replace('/',''))"`
archive_file_name="dss-plugin-${plugin_id}-${plugin_version}.zip"
remote_url=`git config --get remote.origin.url`
last_commit_id=`git rev-parse HEAD`


plugin:
	@echo "[START] Archiving plugin to dist/ folder..."
	@cat plugin.json | json_pp > /dev/null
	@rm -rf dist
	@mkdir dist
	@echo "{\"remote_url\":\"${remote_url}\",\"last_commit_id\":\"${last_commit_id}\"}" > release_info.json
	@git archive -v -9 --format zip -o dist/${archive_file_name} HEAD
	@zip -u dist/${archive_file_name} release_info.json
	@rm release_info.json
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

dev:
	@echo "[START] Archiving plugin to dist/ folder... (dev mode)"
	@cat plugin.json | json_pp > /dev/null
	@mkdir -p dist
	@zip -v -9 dist/${archive_file_name} -r . --exclude "tests/*" "env*" ".*" "*/__pycache__/*"
	@echo "[SUCCESS] Archiving plugin to dist/ folder: Done!"

unit-tests:
	@echo "[START] Running unit tests..."
	@( \
		PYTHON_VERSION=`python3 -V 2>&1 | sed 's/[^0-9]*//g' | cut -c 1,2`; \
		PYTHON_VERSION_IS_CORRECT=`cat code-env/python/desc.json | python3 -c "import sys, json; print(str($$PYTHON_VERSION) in [x[-2:] for x in json.load(sys.stdin)['acceptedPythonInterpreters']]);"`; \
		if ! $$PYTHON_VERSION_IS_CORRECT; then echo "Python version $$PYTHON_VERSION is not in acceptedPythonInterpreters"; exit 1; fi; \
	)
	@( \
		python3 -m venv env/; \
		source env/bin/activate; \
		pip3 install --upgrade pip; \
		pip3 install --no-cache-dir -r tests/python/unit/requirements.txt; \
		pip3 install --no-cache-dir -r code-env/python/spec/requirements.txt; \
		export PYTHONPATH="$(PYTHONPATH):$(PWD)/python-lib"; \
		export DICTIONARY_FOLDER_PATH="$(PWD)/resource/dictionaries"; \
		pytest -o junit_family=xunit2 --junitxml=unit.xml tests/python/unit || true; \
		deactivate; \
	)
	@echo "[SUCCESS] Running unit tests: Done!"

integration-tests:
	@echo "[START] Running integration tests..."
	# TODO add integration tests
	@echo "[SUCCESS] Running integration tests: Done!"

tests: unit-tests integration-tests

dist-clean:
	rm -rf dist

# Docker-based unit tests for Linux environment
# Uses --platform linux/amd64 to ensure consistent behavior on Apple Silicon
#
# Usage:
#   make docker-test-py310              # Without cache (default, CI)
#   make docker-test-py310 USE_CACHE=true  # With pip cache (faster local iteration)

DOCKER_IMAGE_NAME=nlp-ner-test
DOCKER_PLATFORM=linux/amd64
USE_CACHE?=false

define run-docker-test
	@echo "[START] Running unit tests in Docker with Python $(1) (USE_CACHE=$(USE_CACHE))..."
	@docker build \
		--platform $(DOCKER_PLATFORM) \
		--build-arg PYTHON_VERSION=$(1) \
		--build-arg USE_CACHE=$(USE_CACHE) \
		-t $(DOCKER_IMAGE_NAME):py$(1) \
		-f tests/docker/Dockerfile \
		. && \
	docker run --rm --platform $(DOCKER_PLATFORM) $(DOCKER_IMAGE_NAME):py$(1)
	@echo "[DONE] Python $(1) tests completed"
endef

docker-test-py36:
	$(call run-docker-test,3.6)

docker-test-py37:
	$(call run-docker-test,3.7)

docker-test-py38:
	$(call run-docker-test,3.8)

docker-test-py39:
	$(call run-docker-test,3.9)

docker-test-py310:
	$(call run-docker-test,3.10)

docker-test-py311:
	$(call run-docker-test,3.11)

docker-test-py312:
	$(call run-docker-test,3.12)

docker-test-py313:
	$(call run-docker-test,3.13)

# Run all tests with summary (continues on failure, reports at end)
PYTHON_VERSIONS = 3.6 3.7 3.8 3.9 3.10 3.11 3.12 3.13

docker-test-all:
	@failed=""; passed=""; \
	for ver in $(PYTHON_VERSIONS); do \
		echo ""; \
		echo "############################################"; \
		echo "# Testing Python $$ver"; \
		echo "############################################"; \
		target="docker-test-py$$(echo $$ver | tr -d '.')"; \
		if $(MAKE) $$target USE_CACHE=$(USE_CACHE); then \
			passed="$$passed $$ver"; \
		else \
			failed="$$failed $$ver"; \
		fi; \
	done; \
	echo ""; \
	echo "############################################"; \
	echo "#              TEST SUMMARY"; \
	echo "############################################"; \
	if [ -n "$$passed" ]; then echo "PASSED:$$passed"; fi; \
	if [ -n "$$failed" ]; then echo "FAILED:$$failed"; fi; \
	echo "############################################"; \
	if [ -n "$$failed" ]; then exit 1; fi

docker-clean:
	@echo "Removing Docker test images..."
	@docker rmi -f $(DOCKER_IMAGE_NAME):py3.6 $(DOCKER_IMAGE_NAME):py3.7 $(DOCKER_IMAGE_NAME):py3.8 $(DOCKER_IMAGE_NAME):py3.9 $(DOCKER_IMAGE_NAME):py3.10 $(DOCKER_IMAGE_NAME):py3.11 $(DOCKER_IMAGE_NAME):py3.12 $(DOCKER_IMAGE_NAME):py3.13 2>/dev/null || true
	@echo "Docker images cleaned"
