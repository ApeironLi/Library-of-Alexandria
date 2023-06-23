**作者：** ApeironLi，[原文地址](https://www.jianshu.com/p/eb5335708f2a
**修订：** 2023.6.12 Ver.1.0

nvcc是CUDA语言的编译器，将程序编译成可执行的二进制文件。

nvidia-smi(NVIDIA System Management Interface)是一种 命令行实用工具 ，旨在帮助管理和监控NVIDIA-GPU设备。

CUDA有工作接口（runtime api）和 驱动接口（driver api），两者都有对应的CUDA版本。工作接口的支持文件则是由CUDA Toolkit installer安装；驱动接口的支持文件由GPU driver installer 安装。nvcc --version显示的是工作时接口对应的CUDA版本，而 nvidia-smi显示的是驱动接口对应的CUDA版本。

nvcc作为与CUDA Toolkit一起安装的CUDA compiler-driver tool， 只知道它自身构建时的CUDA工作接口版本，并不知道CUDA驱动接口的版本。当然，如果在CUDA Toolkit安装时选择了列表中的同时安装显卡驱动，则可以保证工作接口和运行接口的版本一致，如果单独使用则可能导致版本不一致。

通常， 驱动接口的版本能向下兼容运行接口的版本。当nvcc版本小于nvidia-smi时则不会出现严重问题。