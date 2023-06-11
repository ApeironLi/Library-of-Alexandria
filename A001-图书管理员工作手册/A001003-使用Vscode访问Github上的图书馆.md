## 使用Vscode访问Github上的图书馆

**作者：** ApeironLi

**编号：** A001003

**修订：** 2023.6.11 Ver.1.0

### 摘要

本文旨在记录如何使用Vscode中的Github Pull Requests and Issues extension访问和管理亚历山大图书馆。

### 1. 第一步：安装Git

https://git-scm.com/download

### 2. 第二步：安装Git插件

Github Pull Requests and Issues extension

### 3. 第三步：关联Github账户和Vscode

- 登陆管理员个人的Github账号（左侧边框栏-账户-登陆）
- 左侧边框栏-账户-使用Github登陆以使用Github Pull Requests and Issues extension（此时链接已经成功建立）

### 4. 项目管理

点击左侧边框栏“源代码管理”，可以通过“打开文件夹”打开一个本地源码项目，或通过“克隆仓库”打开一个Github上的源码项目。在源代码管理打开后，左侧栏目上侧源代码管理后的三个点中把“源代码管理存储库”视图打开。

#### 4.1 本地项目上传
- 选择一个本地项目。
- 在“源代码管理存储库-需要选择的项目”的三点扩展中选取“远程-添加远程存储库”，选择待上传的远程Github库，同时为其取一个代号（代号可以不与远程库名一致）
- 为本次上传添加一段简单的commit并点击提交（源代码管理栏目中），再点击同步更改即可。

#### 4.2 克隆仓库并更改

