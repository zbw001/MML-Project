# MML Project

## 准备工作

- 显存要求：>= 48G
- python 版本：3.8
- 运行时需要从网络下载模型，请保持网络通畅

### 安装
```bash
pip install -e .
```

### 下载模型与数据
#### Layout Predictor 模型
下载链接：https://drive.google.com/file/d/1QU0TKFdlNmJPlOReqS41dNU7J8RLztil/view?usp=sharing

请放到 mml_project/layout_predictor/checkpoints

#### 训练数据
下载链接：https://drive.google.com/file/d/13UuBEvbQJFGiYpDRyQ2U1TSRaVfLFp1q/view?usp=sharing

解压到任意文件夹，详见 Layout Predictor 训练部分

## 运行

### Layout Predictor

#### 训练

修改 `mml_project/layout_predictor/configs/data/coco_rel.yaml`，将 `data_dir` 改为数据解压的目录

开始训练：

```bash
cd mml_project/layout_predictor
python trainer.py
```

#### 推理

`mml_project/layout_predictor/layout_predictor.py` 提供了供其他模块调用的接口。详见 `layout_predictor.py` 的 `__main__`。

### Attention Optimization

#### 生成单张图片

```bash
cd mml_project/image_generation
python sampler.py [--config CONFIG] [--prompt PROMPT] [--seed SEED] [--debug]
```

参数解释：

`--config` 指定了配置文件。`--config configs/sd_baseline.yaml` 将运行不作 Cross Attention 控制的 Baseline，`--config configs/replicate.yaml` 将运行我们的复现。`--config configs/ours.yaml` 是我们的改进版本。

`--prompt` 指定了输入的 text prompt。

`--debug` 选项开启时，diffusion 过程的中间图像和 attention map 会被保存，这可能造成较大的磁盘读写。

运行完成后，结果保存至 `mml_project/image_generation/outputs` 文件夹。


#### 生成多张图片

请参考 `mml_project/datasets/gpt4.json`，编写包含将要使用的 prompt 的 json 文件。

```
cd mml_project/image_generation
python evaluate.py [-h] [--config CONFIG] [--dataset DATASET]
```

`--config` 同生成单张图片时的选项
。
`--dataset` 指定了前述 json 文件。

