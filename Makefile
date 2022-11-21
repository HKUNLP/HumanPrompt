dev:
	pip install --upgrade setuptools
	pip install -e .[dev]

check:
	isort -c examples/ humanprompt/ setup.py
	black examples/ humanprompt/ setup.py --check
	flake8 examples/ humanprompt/ setup.py
	mypy humanprompt/ --ignore-missing-imports

test: dev
	pytest tests
