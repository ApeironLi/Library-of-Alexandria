**作者：** ApeironLi; LeafZhang
**修订：** 2023.07.21 Ver.1.3
#Ubuntu

### Basic TODO List
- [ ] 安装Ubuntu18.04系统
- [ ] 安装Nvidia显卡驱动
- [ ] 安装cuda
- [ ] 安装cudnn
- [ ] 安装Anaconda
- [ ] 安装Pytorch

### Advanced TODO List
- [ ] 配置VPN
- [ ] 安装和配置Vscode
- [ ] 配置远程连接
- [ ] 安装Obsidian
- [ ] 安装Google浏览器
- [ ] 安装Zotero
- [ ] 安装搜狗输入法

---
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

#### 1.3 挂载硬盘
- 1.3.1 对于不需要格式化的硬盘，搜索应用disk将硬盘直接挂载（如需要格式化则在disk应用中直接格式化硬盘即可）
- 1.3.2 查看硬盘的UUID
```sh
sudo fdisk -l # 查看硬盘名称，一般是/dev/sda1或/dev/sdb1，注意要在Device栏查看
sudo blkid /dev/sda1 # 查看硬盘UUID，注意不是PTUUID或者PARTUUID
```
- 1.3.3 设置开机自动挂载
```sh
sudo nautilus
进入 /etc/fstab 在文件最后添加
UUID="xxxxx"（查询的UUID） /mnt(挂载点不存在时需要手动创建) ext4 defaults 0 2
```
- 1.3.4 挂载
```sh
sudo mount -a
```
- 1.3.5 重启生效

---
### 2. 安装Nvidia显卡驱动程序

#### 2.1 安装之前
- 2.1.1 显卡驱动程序必须与cuda、pytorch版本匹配。我们推荐使用的cuda版本为11.7，推荐使用的pytorch版本为对应cuda11.7的pytorch版本。因此需要保证安装的显卡驱动版本在[450.80.02以上](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)。可以在[Nvidia官网](https://www.nvidia.cn/Download/index.aspx?lang=cn)查询自己显卡对应的驱动程序版本（选择生产分支、语言为中文简体）。
- 2.1.2 设置root密码：由于后面会用到root，需要先为root赋密码
```sh
sudo passwd
# enter your passwd
```

#### 2.2 安装显卡驱动
- 2.2.0 直接在应用中打开Software&Updates的Additional Drivers，选择版本>450的显驱安装。若能够完成，直接跳到2.2.5。
- 2.2.1 下载显卡驱动（.run程序）（**SmallBreak：下载时间不长不短可以泡杯茶**），下载位置就在2.1.1中Nvidia官网。
- 2.2.2 禁用Ubuntu自带的nouveau开源显卡驱动程序
	- 进入管理员权限的图形化界面：sudo nautilus
	- 创建黑名单文件 /etc/modprobe.d/blacklist-nouveau.conf
	- 在文件中写入 blacklist nouveau 换行 options nouveau modeset=0
	- 执行 sudo update-initramfs -u 和 sudo reboot使更改生效（此时显示屏分辨率可能异常，忽略即可）
	- 检查是否成功禁用 lsmod | grep nouveau（无输出即禁用成功）
- 2.2.3 更新apt,安装依赖库
```sh
sudo apt-get update
sudo apt-get upgrade #时间会比较长
sudo apt intstall gcc
sudo apt install make
sudo apt install libc-dev libc6-dev
```

- 2.2.4 下载驱动程序后，切换到包含驱动程序包的目录并通过以 root 身份运行 sh ./NVIDIA-Linux-x86_64-525.116.04.run 安装驱动程序。
	- 注：32-bit适应性库不需要安装
	- 注：vulkan库不需要安装
	- 注：libglvnd库不需要安装
 	- 注：安装完成后，执行software更新并重新启动
- 2.2.5 检查驱动安装：
```sh
nvidia-smi
```

---
### 3. 安装Cuda-Toolkit

#### 3.1 安装Cuda
- 3.1.0 卸载已有的cuda-toolkit，防止后续出现安装多个cuda的情况。
```sh
sudo apt-get autoremove nvidia-cuda-toolkit
```
- 3.1.1 前往官网选择对应的[Cuda-Toolkit](https://developer.nvidia.com/cuda-11-7-0-download-archive)（我们强烈推荐11.7经典版本）按照提示执行下载指令并切换到包含cuda安装程序包的目录（选择run而不是deb），通过root身份运行sh ./xxxxxxx.run安装cuda-toolkit。
- 3.1.2 安装cuda-toolkit：在安装清单中取消安装驱动（第一项，回车键可以取消打叉X），其余都需要安装。
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
- 3.1.4 检查配置是否成功
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

---
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
官网地址（需要nvidia账号）：https://developer.nvidia.com/rdp/cudnn-download
```
- 4.3 进入deb安装包地址并开始配置：
```sh
sudo dpkg -i xxxxxxxx.deb
sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/ #根据上一行执行提示
sudo apt-get update
# 安装cuda运行库
sudo apt-get install libcudnn8=8.9.2.26-1+cuda11.8 #没有匹配的11.7，使用11.8向下兼容
# 安装cuda开发者库
sudo apt-get install libcudnn8-dev=8.9.2.26-1+cuda11.8
# 安装代码实例
sudo apt-get install libcudnn8-samples=8.9.2.26-1+cuda11.8
```
- 4.4 检查cudnn是否成功安装
```sh
cp -r /usr/src/cudnn_samples_v8/ $HOME
cd $HOME/cudnn_samples_v8/mnistCUDNN
make clean && make
./mnistCUDNN
```
```Output
Test passed!
```

---
### 5. 安装Anaconda
- 5.1 下载Anaconda安装包（ 不要选择清华源）
	- 地址1：官网Anaconda3-2023.03-1-Linux-x86_64.sh（版本可以根据官网提供）
	- 地址2：seuiv@10.193.0.31: /mnt/LOA-Local/Anaconda3-2023.03-1-Linux-x86_64.sh
- 5.2 进入包含安装包的目录，执行安装包下载Anaconda
```sh
bash Anaconda3-2023.03-1-Linux-x86_64.sh #注意enter别把yes/no按掉了
```
- 5.3 重启电脑
- 5.4 更新conda
```conda
conda update conda
```
- 5.5 生成你的第一个conda虚拟环境
```conda
conda create --name [你的环境名字] python==3.x.x[python版本]
```

---
### 6. 安装Pytorch
- 6.0 请进入5.5创建的环境当中
```conda
conda activate [你的环境名字]
```
- 6.1 特别注意：官网提供的torch安装方式并不适合本教程，请使用pip3进行安装
```sh
pip3 install torch torchvision torchaudio
```
- 6.2 检查torch是否可用，cuda是否正常
```python
python
import torch
torch.cuda.is_available() #获得True表明cuda可用
exit()
```

---
### 7. 配置VPN

#### 7.1 摘要
我们推荐使用Clash在Ubuntu上管理VPN，其他设备（如Windows、MacOS、Androrid和IOS等系统）如何配置VPN可以见[N2ray](https://dash.n2ray.dev/service)和 [大机场](https://大机场.shop/#/login)给出的教程。对于Ubuntu上的Clash，关键在于三个文件：
- Clash 配置文件 config.yaml
- Clash 二进制文件 clash.gz
- Clash 国家文件 Country.mmdb
其中后两个文件可以根据本文附带的脚本直接下载安装，在不同主机上通用。config.yaml则需要自己购买订阅。

#### 7.2 获得配置文件
- 7.2.1 [N2ray](https://dash.n2ray.dev/service)购买：注意，N2ray官方给出的Linux一键命令是无法执行的，由于config.yaml无法从N2ray服务器下载。需要在你的windows设备或MacOS设备上下载后找到配置文件（app内就可以定位）。将找到的配置文件拖到Ubuntu的"/etc/clash"文件夹中（地址千万不能错）并将配置文件重命名为config.yaml。
注意：N2ray相对较贵，且流量较少，但速度较快，且不存在自启动错误问题。
- 7.2.2 [大机场](https://大机场.shop/#/login)购买：注意，N2ray官方给出的Linux一键命令是无法执行的，由于config.yaml无法从N2ray服务器下载。需要在你的windows设备或MacOS设备上使用一键配置ClashX后找到配置文件。
注意：大机场非常便宜，且流量很多，但速度相对慢，且存在开机自启动错误，每次开机都需要手动启动。可以在seuiv@10.193.0.31:/mnt/LOA-Local/public-config.yaml获取公用配置文件。

#### 7.3 配置Clash通用文件并设置开机自启动
使用我们提供的脚本clash_install.sh，需要修改其中的若干部分：
\[Service\]中User=主机的用户名。

#### 7.4 其他注意事项
- 7.4.1. 如何检查Clash的运行状态？
```sh
systemctl status clash --no-pager -l
```
- 7.4.2. 如果你使用的是大机场，则需要每次开机后手动执行如下命令开启Clash。
```sh
systemctl start clash
```

---
### 8. 安装和配置Vscode
- 8.1 [官网](https://code.visualstudio.com)下载vscode安装包(.deb文件)，双击deb文件安装
- 8.2 登陆你的Github账户（Optional）

---
### 9. 配置远程连接

#### 9.1 配置ssh远程连接
- 9.1.1 安装OpenSHH用户端（允许控制其他设备）
```sh
sudo apt-get install openssh-client
```
- 9.1.2 安装OpenSHH服务端（允许被其他设备控制）
```sh
sudo apt-get install openssh-server
```
- 9.1.3 在被控端查询ip
```sh
ifconfig
```
- 9.1.4 在控制端为被控端生成密钥（Ubuntu和Macbook）
```sh
ssh-keygen -R 被控端ip
```
- 9.1.5 在控制端连接被控端，注意这种控制方式仅限校园网连接。还需要注意，连接SEU-WLAN和CIAO时主机的IP地址不同。
```sh
ssh seuiv@xxx.xxx.xxx.xxx(用户名@ip地址)
```
- 9.1.6 校外控制主机
	- 9.1.6.1 在控制端下载[东南大学VPN客户端](https://vpn.seu.edu.cn/)
	- 9.1.6.2 连接东大VPN：地址vpn.seu.edu.cn；账号为一卡通号，需要手机验证码。
	- 9.1.6.3 按照9.1.5远程连接即可，注意此时被控端只能连接SEU-WLAN。
- 9.1.7 其他注意事项：远程连接执行程序时需要保持连接不断，因此需要控制端和被控端均处于开机且稳定连接状态。因此建议使用你的移动电脑通过Sunclient远程控制一台本地Ubuntu主机，再使用这台主机控制实验室的server。

#### 9.2 配置Vscode远程连接
- 9.2.1 在Vscode内安装插件 Remote-SSH，Remote Explorer和Remote-SSH:Editing Configuration Files.
- 9.2.2 左下角远程连接按钮->连接到Host->添加新的Host->选择第一个配置文件更新

#### 9.3 [[A001003-使用Vscode访问Github上的图书馆]]

#### 9.4 安装向日葵
下载[图形版Sunclient](https://sunlogin.oray.com)

---
### 10. [[A001002-Obsidian快速入门介绍]]

---
### 11. Google Chrome
直接搜索Google Chrome即可。

---
### 12. 安装Zotero
在Ubuntu应用商店直接安装即可。

---
### 13. 安装输入法（Rime/Google/搜狗）
- 13.0 添加中文语言支持：系统设置->区域和语言->管理已安装的语言->在“语言”tab下点击“添加或删除语言”，弹出“已安装语言”窗口，勾选中文简体，点击应用，等待下载。
rime见13.1，Google见13.2，
- 13.1 注意：在我们的测试中，搜狗输入法出现了一些问题。如果你想要安装轻量化的输入法rime（安装时少折腾，配置时多折腾），请执行完13.1后，直接结束安装输入法的步骤。
所以，如果你想安装rime输入法：
```sh
sudo apt-get install ibus-rime
```
然后在系统设置->区域和语言->输入源中，+号搜索rime，添加即可。输入框中按下F4，选择简体输入法。
关于rime的个性化配置，请自行搜索rime官方文档
- 13.2 注意：在我们的测试中，搜狗输入法出现了一些问题。如果你想要安装轻量化的输入法Google（安装时少折腾，配置时多折腾），请执行完13.2后，直接结束安装输入法的步骤。
所以，如果你想安装Google输入法：
```sh
sudo apt-get install fctix
sudo apt-get install fctix-googlepinyin
```
然后在系统设置->区域和语言->输入源中，+号搜索intelligence，添加即可。同时，切换管理已安装的语言->键盘输入法系统为fctix并重启电脑。
- 13.3 以下为搜狗输入法安装。卸载系统自带的ibus输入法：
```sh
sudo apt purge ibus
```
- 13.4 安装并选择fcitx，安装后在“语言”tab下将键盘输入法系统切换为fcitx，点击将配置应用到整个系统。
```sh
sudo apt-get install fcitx
```
- 13.5 在[搜狗输入法官网](https://shurufa.sogou.com/linux)下载搜狗输入法deb文件
- 13.6 在文件下载位置使用dpkg进行搜狗输入法安装
```sh
sudo dpkg -i sogoupinyin_xxxxx_amd64.deb
```
- 13.7 安装输入法依赖项
```sh
sudo apt install libqt5qml5 libqt5quick5 libqt5quickwidgets5 qml-module-qtquick2
sudo apt install libgsettings-qt1
```
- 13.8 重启电脑
- 13.9 管理输入法，右侧上方小键盘->配置输入法（Configure Current Input Method）->添加搜狗输入法（sogoupinyin）并在Global Config中将Trigger Input Method设置为习惯的切换方法。
