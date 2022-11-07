dev:
	pip install -e .[all]

check:
	isort -c examples/ unifiedhumanprompt/ setup.py
	black examples/ unifiedhumanprompt/ setup.py --check
	flake8 examples/ unifiedhumanprompt/ setup.py
	mypy examples/ unifiedhumanprompt/ setup.py

test:
	pytest tests
