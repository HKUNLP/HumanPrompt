check:
	isort -c configs/ examples/ unifiedhumanprompt/ setup.py
	black configs/ examples/ unifiedhumanprompt/ setup.py --check
	flake8 configs/ examples/ unifiedhumanprompt/ setup.py
	mypy configs/ examples/ unifiedhumanprompt/ setup.py
