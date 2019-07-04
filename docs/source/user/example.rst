======
大标题
======

.. note::
    中文标题需要符号的数量至少是中文字数的两倍

.. warning::
    符号的数量只可以多，不可以少。

小标题1
###########

小标题2
*********

小标题3(正常使用)
========================

小标题4
-------------------

推荐使用大标题、小标题3和小标题4

官方文档 http://docutils.sourceforge.net/docs/user/rst/quickref.html

`熟悉markdown的同学推荐参考这篇文章 <https://macplay.github.io/posts/cong-markdown-dao-restructuredtext/#id30>`_

\<\>内表示的是链接地址，\<\>外的是显示到外面的文字

常见语法
============

*emphasis*

**strong**

`text`

``inline literal``

http://docutils.sf.net/ 孤立的网址会自动生成链接

显示为特定的文字的链接 `sohu <http://www.sohu.com>`_

突出显示的
    上面文字

正常缩进

    形成锻炼



特殊模块
============

选项会自动识别

-v           An option
-o file      Same with value
--delta      A long option
--delta=len  Same with value


图片

.. image:: ../figures/procedures.PNG
    :height: 200
    :width: 560
    :scale: 50
    :alt: alternate text
    :align: center

显示一个冒号的代码块::

    中间要空一行

::

    不显示冒号的代码块

.. code-block:: python

    :linenos:
    :emphasize-lines: 1,3

    print("专业的代码块")
    print("")
    print("有行号和高亮")

数学块
==========

.. math::

    H_2O + Na = NaOH + H_2 \uparrow

复杂表格
==========

+------------------------+------------+----------+----------+
| Header row, column 1   | Header 2   | Header 3 | Header 4 |
| (header rows optional) |            |          |          |
+========================+============+==========+==========+
| body row 1, column 1   | column 2   | column 3 | column 4 |
+------------------------+------------+----------+----------+
| body row 2             | Cells may span columns.          |
+------------------------+------------+---------------------+
| body row 3             | Cells may  | - Table cells       |
+------------------------+ span rows. | - contain           |
| body row 4             |            | - body elements.    |
+------------------------+------------+---------------------+

简易表格
==========

=====  =====  ======
   Inputs     Output
------------  ------
  A      B    A or B
=====  =====  ======
False  False  False
True   True   True
=====  =====  ======

[重要]各种链接
===================

各种链接帮助我们连接到fastNLP文档的各个位置

\<\>内表示的是链接地址，\<\>外的是显示到外面的文字

:doc:`根据文件名链接 </user/with_fitlog>`

:mod:`~fastNLP.core.batch`

:class:`~fastNLP.Batch`

~表示只显示最后一项

:meth:`fastNLP.DataSet.apply`

