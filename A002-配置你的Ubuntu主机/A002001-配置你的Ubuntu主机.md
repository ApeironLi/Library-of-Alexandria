## A002001：配置你的Ubuntu主机

**作者：** ApeironLi

**编号：** A002001

**修订：** 2023.6.12 Ver.1.0

### Basic TODO List
- [ ] 安装Ubuntu18.04系统
- [ ] 安装Nvidia显卡驱动
- [ ] 安装cuda
- [ ] 安装cudnn
- [ ] 安装Anaconda
- [ ] 安装Pytorch
- [ ] 安装Vscode
- [ ] 测试脚本检查
### Advanced TODO List
- [ ] 安装输入法
- [ ] 配置VPN
- [ ] 安装浏览器
- [ ] 配置Vscode远程控制

### 1. 安装/重装Ubuntu系统

#### 1.1 制作系统盘

- 1.1.1 拷贝系统盘文件，我们采用最稳定的Ubuntu18.04.6作为推荐系统：
```sh
seuiv@10.193.0.31: /mnt/LOA-Local/ubuntu-18.04.6-desktop-amd64.iso
```
- 1.1.2 找到一个U盘（内存大于3G），备份其中的所有内容。
- 1.1.3 找到一台可用的Ubuntu电脑，使用自带的Startup Disk Creator制作启动盘（插入U盘后U盘中原有内容将被清空）。在Source disc image中选择1.1.1中下载的iso文件作为源iso文件。选择插入的U盘并制作启动盘。

#### 1.2 安装/重装系统
- 特别提醒：如果需要重装，我们强烈建议提前将所有文件都备份到挂载硬盘上。
- 1.2.1 重启需要重装系统的电脑，开始重启后按F2键进入[[A002002-BIOS模式]]。选择引导程序为U盘中的Ubuntu系统安装程序：选择BIOS功能，将启动优先顺序设置为你的U盘（是否包含UEFI均可）。最后选择储存并离开设定。
- 1.2.2 在安装界面中选择Install Ubuntu进入系统安装流程。
- 1.2.3 选择偏好语言（我们强烈推荐English(US)）
- 1.2.4 连接WIFI 
- 1.2.5 选择“普通安装”和“安装Ubuntu时下载更新”
- 1.2.6 我们推荐使用Erase disk and install Ubuntu来进行重新安装，但注意，这种方式会清除所有原系统上的文件（但不会清除挂载硬盘上的文件）。
- 1.2.7 选择系统安装的硬盘，不要选择将系统装在挂载硬盘上（即保持默认即可）。
- 1.2.8 选择你的时区、名称和密码。本组所有名称一律统一为seuiv以便远程控制。
---
##### BreakTime：接下来需要等待一段相对长的时间，可以去吃个饭

### 2. 安装Nvidia显卡驱动程序

#### 2.1 安装之前
- 2.1.1 显卡驱动程序必须与cuda、pytorch版本匹配。我们推荐使用的cuda版本为11.7，推荐使用的pytorch版本为对应cuda11.7的pytorch版本。因此需要保证安装的显卡驱动版本在[450.80.02以上](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。可以在[Nvidia官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)查询自己显卡对应的驱动程序版本（选择生产分支、语言为中文简体）。
- 2.1.2 设置root密码：由于后面会用到root，需要先为root赋密码
```sh
sudo passwd
# enter your passwd
```


#### 2.2 安装显卡驱动
- 2.2.1 下载显卡驱动（.run程序）（**SmallBreak：下载时间不长不短可以泡杯茶**）
- 2.2.2 禁用Ubuntu自带的nouveau开源显卡驱动程序
	- 进入管理员权限的图形化界面：sudo nautilus
	- 创建黑名单文件 /etc/modprobe.d/blacklist-nouveau.conf
	- 在文件中写入 blacklist nouveau 换行 options nouveau modeset=0
	- 执行 sudo update-initramfs -u 和 sudo reboot使更改生效
	- 检查是否成功禁用 lsmod | grep nouveau（无输出即禁用成功）
- 2.2.3 安装依赖库
```sh
sudo apt intstall gcc
sudo apt install make
```
- 2.2.4 下载驱动程序后，切换到包含驱动程序包的目录并通过以 root 身份运行 sh ./NVIDIA-Linux-x86_64-525.116.04.run 安装驱动程序。
	- 注：32-bit适应性库不需要安装
	- 注：vulkan库不需要安装
	- 注：libglvnd库不需要安装
- 2.2.5 检查驱动安装：
```sh
nvidia-smi
```

### 3. 安装Cuda-Toolkit

#### 3.1 安装Cuda
- 3.1.1 前往官网选择对应的[Cuda-Toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive)（我们强烈推荐11.7经典版本）按照提示执行下载指令并切换到包含cuda安装程序包的目录，通过root身份运行sh ./xxxxxxx.run安装cuda-toolkit。
---
##### BreakTime：接下来需要等待一段相对更长的时间，可以去上节课
- 3.1.2 安装cuda-toolkit：在安装清单中取消安装驱动（第一项），其余都需要安装。
- 3.1.3 为cuda-toolkit添加配置文件
管理员权限打开可视化界面
```sh
sudo nautilus
```
进入地址：/home/.bashrc
添加：
```sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export PATH=$PATH:/usr/local/cuda/bin
export CUDA_HOME=$CUDA_HOME:/usr/local/cuda
```
刷新配置：
```sh
source ~/.bashrc
```
检查配置是否成功：
```sh
nvcc -V
```

#### 3.2 测试cuda是否安装成功

- 3.2.1 运行同步下载的demo程序：
```sh
/usr/local/cuda/extras/demo_suite/deviceQuery
```
- 3.2.2 运行nvidia-smi
- 3.2.3 运行nvcc -V
注：[[A002003-nvcc和nvidia-smi显示的cuda版本不同]]？

### 4. 安装Cudnn

Cudnn可以用很多方式安装，我们推荐基于deb安装包的安装方式。

- 4.1 安装依赖库
```sh
sudo apt install zlib1g
sudo apt install g++
sudo apt install libfreeimage3 libfreeimage-dev
```

- 4.2 下载Cudnn的deb安装包，地址：
```sh
seuiv@10.193.0.31: /mnt/LOA-Local/cudnn-local-repo-ubuntu1804-8.9.2.26_1.0-1_amd64.deb
```

- 4.3 进入deb安装包地址并开始配置：
```sh
sudo dpkg -i cudnn-local-repo-${distro}-8.9.2.26-1_1.0-1_amd64.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
# 安装cuda运行库
sudo apt-get install libcudnn8=8.9.2.26-1+cuda11.8 #没有匹配的11.7，使用11.8向下兼容
# 安装cuda开发者库
sudo apt-get install libcudnn8-dev=8.9.2.26-1+cuda11.8
# 安装代码实例
sudo apt-get install libcudnn8-samples=8.9.2.26-1+cuda11.8
```

- 4.5 检查cudnn是否成功安装
```sh
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```
```Output
Test passed!
```
