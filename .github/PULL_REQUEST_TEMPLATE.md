Description：简要描述这次PR的内容

Main reason: 做出这次修改的原因

Checklist  检查下面各项是否完成

Please feel free to remove inapplicable items for your PR.

-	[ ] The PR title starts with [$CATEGORY] (such as [Models], [Modules], [Core], [io], [Doc], 分别对应各个子模块)
-	[ ] Changes are complete (i.e. I finished coding on this PR)  代码写完了
-	[ ] All changes have test coverage  修改的地方经过测试。对于可复用部分的修改，例如core/和modules/，测试代码必须提供。其他部分建议提供。
-	[ ] Code is well-documented  注释写好，文档会从注释中自动抽取
-	[ ] To the my best knowledge, examples are either not affected by this change, or have been fixed to be compatible with this change  这种情况请找核心开发人员

Changes: 逐项描述修改的内容
-	Switch to sparse_coo_matrix for torch v1.0. #282
- Fix bug that nx graph to dgl graph is not properly converted. #286
