PYTHON3        = python3
SOURCEDIR      = qoala
TESTDIR        = tests
EXAMPLEDIR     = examples
RUNEXAMPLES    = ${EXAMPLEDIR}/run_examples.py
PIP_FLAGS      = --extra-index-url=https://${NETSQUIDPYPI_USER}:${NETSQUIDPYPI_PWD}@pypi.netsquid.org

help:
	@echo "install           Installs the package (editable)."
	@echo "verify            Verifies the installation, runs the linter and tests."
	@echo "tests             Runs the tests."
	@echo "examples          Runs the examples and makes sure they work."
	@echo "lint              Runs the linter."
	@echo "docs              Creates the html documentation"
	@echo "clean             Removes all .pyc files."

_check_variables:
ifndef NETSQUIDPYPI_USER
	$(error Set the environment variable NETSQUIDPYPI_USER before installing)
endif
ifndef NETSQUIDPYPI_PWD
	$(error Set the environment variable NETSQUIDPYPI_PWD before installing)
endif

clean:
	@/usr/bin/find . -name '*.pyc' -delete

lint-isort:
	$(info Running isort...)
	@$(PYTHON3) -m isort --check --diff ${SOURCEDIR} ${TESTDIR} ${EXAMPLEDIR}

lint-black:
	$(info Running black...)
	@$(PYTHON3) -m black --check ${SOURCEDIR} ${TESTDIR} ${EXAMPLEDIR}

lint-flake8:
	$(info Running flake8...)
	@$(PYTHON3) -m flake8 ${SOURCEDIR} ${TESTDIR} ${EXAMPLEDIR}


lint: lint-isort lint-black lint-flake8

mypy:
	@$(PYTHON3) -m mypy ${SOURCEDIR}

all-tests:
	coverage run -m pytest tests -n auto

unit-tests:
	coverage run -m pytest tests --ignore=tests/integration

integration-tests:
	coverage run -m pytest tests/integration

test-report:
	coverage report --omit="tests/*"

examples:
	@$(PYTHON3) ${RUNEXAMPLES}

docs html:
	@${MAKE} -C docs html

install: _check_variables
	@$(PYTHON3) -m pip install -e . ${PIP_FLAGS}

install-dev: _check_variables
	@$(PYTHON3) -m pip install -e .[dev] ${PIP_FLAGS}

verify: clean lint mypy all-tests examples _verified

_verified:
	@echo "Everything works!"

.PHONY: clean lint tests verify install examples docs
