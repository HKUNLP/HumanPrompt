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
- [ ] Shorten the repo name to be published on PyPI
- [ ] Add methods(ReAct, AMA_prompting, self ask) that requires external API
- [ ] Add one program generation method(prompt prog)
- [ ] Add one data generation method(ZeroGenI)
- [ ] Add one skg method(Binder)
- [ ] Init the experiments part
- [ ] Support batch running function
- [ ] Start UI construction
- [ ] Add more tests

## Completed ✓
- [x] Add CLI
- [x] Add static checkings
- [x] Derive the manifest from the method, make full use of its init, run and run_batch(In progress)
- [x] Support in-context examples when using method.run
- [x] Path and preparation for PyPI
- [x] Add AutoMethod Factory
