# 快速入门 fastNLP 文档编写

本教程为 fastNLP 文档编写者创建，文档编写者包括合作开发人员和文档维护人员。您在一般情况下属于前者，
只需要了解整个框架的部分内容即可。

## 合作开发人员

FastNLP的文档使用基于[reStructuredText标记语言](http://docutils.sourceforge.net/rst.html)的
[Sphinx](http://sphinx.pocoo.org/)工具生成，由[Read the Docs](https://readthedocs.org/)网站自动维护生成。
一般开发者只要编写符合reStructuredText语法规范的文档并通过[PR](https://help.github.com/en/articles/about-pull-requests)，
就可以为fastNLP的文档贡献一份力量。

如果你想在本地编译文档并进行大段文档的编写，您需要安装Sphinx工具以及sphinx-rtd-theme主题。然后在本目录下执行`make dev` 命令，
并在浏览器访问 http://0.0.0.0:8000/ 查看文档。 该命令只支持Linux和MacOS系统，在结束查看后需按 Control(Ctrl) + C 退出。
如果你在远程服务器尚进行工作，您可以通过浏览器访问 http://{服务器的ip地址}:8000/ 查看文档，但必须保证服务器的8000端口是开放的。
如果您的电脑或远程服务器的8000端口被占用，程序会顺延使用8001、8002……等端口，具体以命令行输出的信息为准。

我们在[这里](./source/user/example.rst)列举了fastNLP文档经常用到的reStructuredText语法（网页查看请结合Raw模式），
您可以通过阅读它进行快速上手。FastNLP大部分的文档都是写在代码中通过Sphinx工具进行抽取生成的，
您还可以参考这篇[未完成的文章](./source/user/docs_in_code.rst)了解代码内文档编写的规范。

## 文档维护人员

文档维护人员需要了解 Makefile 中全部命令的含义，并了解到目前的文档结构
是在 sphinx-apidoc 自动抽取的基础上进行手动修改得到的。
文档维护人员应进一步提升整个框架的自动化程度，并监督合作开发人员不要破坏文档项目的整体结构。