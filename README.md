#### 配置共享文件夹

在共享文件设置中关联主机的的共享文件夹，选择固定分配，具体步骤如下

1.共享文件夹的配置

![avatar](./1542617455304.jpg)

2.开启ubuntu的增强功能 - devices->insert geuset xxx

3.

```
sudo apt install nfs-common
sudo apt install cifs-uitls
sudo mount -t vboxsf <macShare>  <ubuntuShare>
```





#### ubuntu安装python
sudo apt install python-pip python-dev

配置python虚拟开发环境

```
pip install virtualenv

virtualenv --system-site-packages -p python2.7  ./venv #创建一个虚拟环境

source ./bin/active #进入虚拟环境

deactivate #退出
```

ubuntu安卓的python不带tkinter库，在使用matplotlib会出错

因为tkinter不是第三方库，因此不能用pip安装，通过下面命令来安装

```
sudo apt install python-tk
```

#### python 安装tensorflow

```
pip install --upgrade pip
pip list

pip install --upgrade tensorflow 
pip install --user --upgrade tesorflow （该命令即使在虚拟环境下也会安装的home目录）
```

#### tensorflow
tensorflow 的架构设计为client - service模式
client负责图的构建，图由tensor和operation构成，
service:Session，run运行客户端构建的图，这个计算过程可以是GPU，CPU，TPC来运行，并且支持分布式。

