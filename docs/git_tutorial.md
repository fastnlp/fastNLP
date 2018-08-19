# 怎样使用Git进行开发
具体的开发流程为
- 在github网页，fork仓库
- git clone 下载仓库
- git branch 新建分支
- git add/commit 提交修改
- git pull 从远程下载到本地，让本地保持与远程相同的最新状态
- git merge/rebase 分支远程和本地的相应合并，为`git push`做准备
- git push 将本地更改上传到远程仓库
- 在github网页，创建pull request，等待code review后接收修改



## fork 远程仓库
如果要对某个项目进行开发，但又没有对远程仓库的写权限，就必须fork这个仓库，然后使用pull request对项目进行修改。
进入github上的远程仓库主页，如fastnlp，点击fork，将整个项目复制到你的github账户下。

## 下载repo
fork项目后，使用`git clone`将项目下载到本地

    git clone [url-of-repo] [name]
将仓库下载到本地，并改名为`name`

## 查看仓库的状态
`clone`成功后，可以在项目中查看项目的状态

    git status
可以看到当前所在分支，已修改，暂存(staged)文件等信息

    git log
可以查看当前分支的历史提交记录

    gitk
图形界面的git，便于查看各分支与提交信息

## 分支切换
每个项目一般会有多个分支，保存项目的多个版本，其中最重要的分支是`master`，是项目的主分支。而开发时一般不直接修改主分支，而是新建一个分支，修改完成后再合并到主分支。

    git branch
查看现有的本地分支，以及当前所在分支

    git branch [name]
基于所在的分支，创建一个分支

    git branch --remote
查看所有远程仓库的分支

    git checkout [branch]
可以切换当前分支到`branch`

    git checkout -b [branch]
创建分支，名为`branch`，并切换过去

## 更改和提交
当切换到一个开发的分支后，就可以修改代码，修改完成后，需要提交你的修改

    git add .
    git commit -m "...."

可以将修改过的文件全部放入暂存区并提交。

    git add [file-name]
    git add -i
    git commit

如果不想将所有修改过的文件都提交，而只提交部分文件。可以在`git add`时指定文件名，或设置`-i`参数，交互式的将文件暂存，再`commit`

    git commit -a
如果想修改最近一次提交，使用`-a`将重新执行最近一次提交。(建议不要修改已经`push`到公开仓库的提交，特别是已经被其他人`pull`到本地的提交)

## 远程管理
一般而言，`git clone`的项目有一个默认的远程仓库`origin`，地址为`clone`时指定的url。

    git remote -v
可以查看当前本地仓库对应的所有远程仓库和地址

    git remote add [name] [url]
增加远程仓库，命名为`name`, 地址为`url`

    git fetch [remote]
从远程仓库`name`拉取更新，拉取之后，可以使用`git branch --remote`和`git checkout`等命令查看远程仓库的分支和切换分支

    git pull [remote] [remote-branch]:[local-branch]
相当于`git fetch`之后执行`git merge`命令，从远程拉取更新后自动将远程分支`remote-branch`合并到本地分支`local-branch`

    git push [remote-repo] [local-branch]:[remote-branch]
将本地分支`local-branch`推送到远程分支`remote-branch`，如果分支不存在，自动创建远程分支。

在开发完成后，我们可能在自定义的分支提交了几次，之后准备将代码`push`上传github，再创建pull request合并到项目主分支。而在运行`git push`之前，我们需要与远程分支同步一次，因为此时可能主分支有新的提交。

    git pull [remote] master:[branch]
可以将远程master分支与本地branch合并，相当于`git merge`操作。

    git pull --rebase [remote] master:[branch]
使用`rebase`模式将远程master分支与本地branch合并，相当于`git rebase`操作。

同步完成后，可以使用`git push`进行提交，这时创建的pull request就绝对没有冲突了。


## 分支修改合并
    git checkout A
    git merge B
可以将`B`分支合并到`A`分支，如果`A`和`B`分支是分叉的，会自动commit一个提交，之后`B`分支消失。

    git checkout B
    git rebase A
可以将`B`中提交合并到`A`分支提交的末尾，不产生新的提交，之后`B`分支消失。需要注意`merge`和`rebase`合并方向相反。

## 创建PR
创建pull request，并设置reviewers，让仓库的管理人看到你的代码，并将你的贡献合并到主分支中。
pull request本质上是并入分支和被并入分支的两个分支进行`merge`操作，所以如果pull request有冲突，可以使用`git merge`解决冲突后重新提交。如果后续需要修改已经pull request的代码，可以直接提交的并入分支，修改后的提交会自动出现在pull request当中。
