dev:
	pip install -e .[all]

check:
	isort -c examples/ humanprompt/ setup.py
	black examples/ humanprompt/ setup.py --check
	flake8 examples/ humanprompt/ setup.py
	mypy humanprompt/ --ignore-missing-imports --scripts-are-modules

test:
	pytest tests
