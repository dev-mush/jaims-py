build:
	rm -rf jaims_py.egg-info dist build
	python setup.py sdist bdist_wheel

deploy: build
	twine upload dist/*
