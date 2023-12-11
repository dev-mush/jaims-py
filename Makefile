build-package:
	rm -rf jaims_py.egg-info dist build
	python setup.py sdist bdist_wheel

deploy: build-package
	twine upload dist/*

clean:
	find . -name '__pycache__' | xargs rm -rf
	find . -name '*.pyc' | xargs rm -rf
	rm -rf jaims_py.egg-info dist build
