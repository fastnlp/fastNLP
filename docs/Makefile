# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXAPIDOC  = sphinx-apidoc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = fastNLP
SPHINXEXCLUDE = ../fastNLP/transformers/*
SOURCEDIR     = source
BUILDDIR      = build
PORT          = 8000

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

apidoc:
	$(SPHINXAPIDOC) -efM -o source ../$(SPHINXPROJ) $(SPHINXEXCLUDE)

server:
	cd build/html && python -m http.server $(PORT)

delete:
	rm -f source/$(SPHINXPROJ).* source/modules.rst && rm -rf build

web:
	make html && make server

dev:
	make delete && make apidoc  && make html && make server

versions:
	sphinx-multiversion "$(SOURCEDIR)" "$(BUILDDIR)" && cd build && python -m http.server $(PORT)

prod:
	make apidoc && make html 

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)