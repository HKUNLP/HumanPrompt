# UnifiedHumanPrompt_init🫡

## Development
~~~
pip install -e . # install unifiedhumanprompt abd soft link configs to unifiedhumanprompt/artifacts/configs
~~~

~~~
pip install pre-commit
pre-commit install # install all hooks
pre-commit run --all-files # trigger all hooks
~~~

## TODOs
- [ ] Add GitHub Actions
- [ ] SHorten the repo name to be published on PyPI
- [ ] Add one program generation method
- [ ] Add one data generation method
- [ ] Add one skg method
- [ ] Init the experiments part
- [ ] Support batch running function

## Completed ✓
- [x] Derive the manifest from the method, make full use of its init, run and run_batch(In progress)
- [x] Support in-context examples when using method.run
- [x] Path and preparation for PyPI
- [x] Add AutoMethod Factory
