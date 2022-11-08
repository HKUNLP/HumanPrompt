dev:
	pip install -e .[all]

check:
	isort -c examples/ unifiedhumanprompt/ setup.py
	black examples/ unifiedhumanprompt/ setup.py --check
	flake8 examples/ unifiedhumanprompt/ setup.py
	mypy unifiedhumanprompt/ --ignore-missing-imports --scripts-are-modules

test:
	pytest tests
