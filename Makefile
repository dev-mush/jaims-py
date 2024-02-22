build-package:
	if [ -d "dist" ]; then rm -rf dist; fi
	if [ -d "build" ]; then rm -rf build; fi
	if [ -d "jaims_py.egg-info" ]; then rm -rf jaims_py.egg-info; fi
	python setup.py sdist bdist_wheel

deploy: build-package
	twine upload dist/*
