Description：简要描述这次PR的内容

Main reason: 做出这次修改的原因


Checklist  检查下面各项是否完成

Please feel free to remove inapplicable items for your PR.

-	[ ] The PR title starts with [$CATEGORY] (例如[bugfix]修复bug，[new]添加新功能，[test]修改测试，[rm]删除旧代码)
-	[ ] Changes are complete (i.e. I finished coding on this PR)  修改完成才提PR
-	[ ] All changes have test coverage  修改的部分顺利通过测试。对于fastnlp/fastnlp/*的修改，测试代码**必须**提供在fastnlp/test/*。
-	[ ] Code is well-documented  注释写好，API文档会从注释中抽取
-	[ ] To the my best knowledge, examples are either not affected by this change, or have been fixed to be compatible with this change  修改导致例子或tutorial有变化，请找核心开发人员

Changes: 逐项描述修改的内容
- 添加了新模型；用于句子分类的CNN，来自Yoon Kim的Convolutional Neural Networks for Sentence Classification
- 修改dataset.py中过时的和不合规则的注释 #286
- 添加对var-LSTM的测试代码

Mention: 找人review你的PR

@修改过这个文件的人
@核心开发人员
