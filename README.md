# MCRA-python
MCRA+OMLSA python version

## 特点
1. 包含MCRA和IMCRA算法过程，按照原版论文进行完美复现
2. python版本，github上独特的存在
3. 流式推理过程，你可以把函数复制到音频流中直接可用，或者用于C函数校对数据
4. 所有函数需要的参数放置在函数旁边，免去二次开发的烦恼

## 使用方法
1. 修改最下方`__main__`函数里面的`sf.read`文件读取路径以及`sf.write`文件写入路径
2. 修改IMCRA_open标志位为True或False
3. 运行代码实现推理，当使用MCRA时会显示plt图片用于验证

## 后续计划
1. 加入设备推理过程，编译成可执行文件之后选择本地设备即可完成降噪，更加方便软件部署
2. 将各个函数分在不同文件上，方便移植和独立验证