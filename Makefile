.PHONY: all install dev-install uninstall clean

all: install

install: clean
	pip install .

dev-install: clean
	pip install -e .

uninstall: clean
	pip uninstall genetic_snake 
	$(RM) genetic_snake_C.cpython-39-x86_64-linux-gnu.so

clean:
	$(RM) -rf __pycache__
	$(RM) -rf genetic_snake/__pycache__
	$(RM) -rf genetic_snake/genetic/__pycache__
	$(RM) -rf genetic_snake/gui/__pycache__
	$(RM) -rf build
	$(RM) -rf genetic_snake.egg-info
	$(RM) -rf *.so
	$(RM) -rf dist/