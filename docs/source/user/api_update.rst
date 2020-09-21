===========================
API变动列表
===========================

2020.4.14
========================

修改了 :class:`fastNLP.core.callback.ControlC` 的 API。

原来的参数 ``quit_all`` 修改为 ``quit_and_do`` ，仍然接收一个 bool 值。新增可选参数 ``action`` ，接收一个待执行的函数，
在 ``quit_and_do`` 的值为 ``True`` 时，退出训练过程后执行该函数。 ``action`` 的默认值是退出整个程序，与原有功能一致。

.. note::
    原有用法 `ControlC(True)` 和 `ControlC(False)` 均可以继续正确执行，但 `ControlC(quit_all=True/False)` 需要修改为
    `ControlC(quit_and_do=True/False)`  。