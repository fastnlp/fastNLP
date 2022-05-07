# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXAPIDOC  = sphinx-apidoc
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = fastNLP
SPHINXEXCLUDE = ../fastNLP/transformers/* ../fastNLP/modules/* ../fastNLP/core/drivers/torch_paddle_driver/* ../fastNLP/core/utils/torch_paddle_utils.py
SOURCEDIR     = source
BUILDDIR      = build
PORT          = 9000

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

apidoc:
	$(SPHINXAPIDOC) -efM -d 6 -o source ../$(SPHINXPROJ) $(SPHINXEXCLUDE)

server:
	cd build/html && python -m http.server $(PORT)

delete:
	rm -f source/$(SPHINXPROJ).* source/modules.rst && rm -rf build

dev:
	make delete && make apidoc  && make html && make server

prod:
	make apidoc && make html 

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)