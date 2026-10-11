# PSDformer：PyTorch 独立参考实现

对应稿件：**Long-Term Stress Conditions Assessment and Prediction of Drivers Based on a Hybrid Deep Learning Framework and Novel Sensor System**。

> 本包是依据所提供稿件编写并测试的独立参考实现，不是作者原始完整代码。作者公开仓库目前是未完成的简化框架，尚未实现稿件描述的完整流程。原始驾驶员数据、标签协议及若干模型参数没有提供，因此本包不声称复现论文 95.1% 的准确率。

## 1. 安装

建议新建 Python 3.11 环境，不要在旧 TensorFlow / Python 3.7 环境中运行。

```bash
conda create -n psdformer python=3.11 -y
conda activate psdformer
cd PSDformer
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

GPU 版请通过 [PyTorch 官方安装页面](https://pytorch.org/get-started/locally/)选择合适版本。本次交付核验使用 Linux CPU；不能把它视为 Windows 或 CUDA 实测。

## 2. 先运行演示

```bash
python -m psdformer demo --output runs/demo --device cpu
python -m unittest discover -s tests -v
```

演示数据明确标为合成数据，只用于验证训练、验证集选模、独立受试者测试、保存模型和导出预测是否能完整执行。合成分类标签及目标不代表真实驾驶员压力的测量协议。

## 3. 核心结构与源码

- `psdformer/blocks.py`：MDSW 分段与二维位置编码、时间注意力、路由器跨通道注意力、分段合并、多尺度解码和密集卷积组件。
- `psdformer/model.py`：PSDformer 配置与完整前向路径，连续预测和可选分类输出。
- `psdformer/data.py`：数据格式检查、受试者划分、训练集归一化。
- 数据处理、损失、指标、训练、评估及预测入口分别放在对应模块中。
- `configs/paper.json`：保留论文明确的 lookback=24、Adam、学习率 0.001、batch size=32；未公布参数使用已注明的可修改参考值。
- `configs/smoke.json`：用于快速运行检查的小模型配置。
- `docs/METHOD_AND_ASSUMPTIONS.md`：逐模块公式映射、作者仓库核查和实现取舍。
- `QA_REPORT.md`、`docs/VALIDATION.md`：实际执行的测试与尚未验证的内容。

模型输入按 `[批量, 历史时间点, 通道]` 排列；连续预测按 `[批量, 未来时间点, 输出变量]` 排列。时间维和通道维始终区分处理。

## 4. 真实数据必须满足的条件

每个窗口都要有明确的受试者 ID。训练、验证和测试按受试者划分，同一人的所有窗口不能同时进入不同集合。输入和连续目标的标准化参数只在训练集拟合；评估时恢复原始目标单位再计算误差。

- 未来目标必须来自历史窗口结束之后，不能混入模型输入。
- 连续压力值和离散压力类别必须由真实的标注方案提供，不能随意相互转换。
- EHS、RHS 等传感器数据的实际校准/湿度补偿系数未给出，不在代码里伪造设备标定。
- 公共 BCG 数据不是现成的带压力标签驾驶数据；本包不会把高血压分组自动改写成压力类别。
- 缺失输入的掩码是工程性健壮处理，不是论文额外提出的缺失模态算法。

具体数据字段和命令见下方数据与运行说明。

## 5. 科学复现边界

1. 论文的部分图和正文不是同一种架构。本实现优先落实正文公式 (1)–(19) 与算法的分段、双阶段注意力、层级编码解码和密集卷积流程。
2. DGC 虽被称为“图卷积”，但正文没有给出图邻接矩阵或图消息传递算子，只给出密集连接的卷积/池化。这里实现已描述的操作，不额外捏造图模型。
3. 原文缺少分类头、类别数、标签定义、完整损失配置及全部网络宽度等细节。可选分类头和相应参数是明确标注的参考扩展。
4. 当前运行测试只证明受控输入下的软件行为，不证明真实驾驶场景中的泛化、可靠性或安全性。
5. 预测和分类结果不用于医疗诊断或自动驾驶安全决策。

## 6. 数据格式

使用 `numpy.savez_compressed` 保存 NPZ，不要使用 object 数组或 pickle。

| 字段 | 格式 |
|---|---|
| `x` | float32 `[N,T,D]`，过去 T 个时间点的 D 通道输入 |
| `subject_ids` | `[N]` 字符串或整数，受试者身份，用于隔离划分 |
| `mask` | 可选 bool `[N,T,D]`，True 表示已观测；非有限输入自动视为缺失 |
| `target` | 连续预测任务所需 float32 `[N,K,O]`，未来 K 步、O 个目标变量；缺失目标用 NaN |
| `y` | 分类任务所需 int `[N]`，类别 0 到 C-1；缺失类别用 -1 |

T 对应配置 `input_len`，D 对应 `input_dim`，K 对应 `pred_len`，O 对应 `output_dim`，C 对应 `num_classes`。默认 5 通道只是稿件 Dataset A 的变量数；具体通道名及顺序由你的数据和配置保持一致。更换时间步长不会自动保持生理时间跨度，需要同步核对采样单位。

可选择 `forecast`、`classification` 或 `joint` 任务。连续预测使用有效目标上的 MSE，分类使用有效标签上的交叉熵；联合训练权重是参考实现设置。分类不是通过阈值化模型的连续输出伪造出来的。

分类标签以输入窗口最后一个时刻为对齐基准；连续目标位于输入窗口之后。若你的真实实验协议要求“未来类别”或其他标签位置，需要明确改变数据对齐，不能直接混用。

## 7. 指标说明

预测指标在逆标准化后的真实目标单位计算，并跳过缺失目标。

- MAE：平均绝对误差。
- MSE：平均平方误差。
- RMSE：MSE 的平方根。
- RSE：遵循稿件式子的 `sqrt(SSE / (n - 2))`；它不是 RMSE，也不是按真实信号标准差归一化的误差。有效点数不足时无定义。
- R²：`1 - SSE/SST`；真实目标恒定时无定义，不伪造为 1。
- 分类指标包含准确率、宏平均 Precision/Recall/F1 和混淆矩阵。

多输出汇总指标受目标的单位和尺度影响，应结合各输出变量分别评估。本包不会修改误差数值去匹配论文图表。

## 8. Windows 启动文件

环境安装并激活后，可以运行 `run_demo.bat`。它只执行 CPU 合成演示并保留控制台，不会自动安装软件。Linux/macOS 可运行 `bash run_demo.sh`。批处理文件随包提供，但本次没有 Windows 实机测试。

## 9. 训练、测试与导出命令

```bash
python -m psdformer train --data data/windows.npz --config configs/paper.json --output runs/experiment --device cpu
python -m psdformer evaluate --data data/windows.npz --checkpoint runs/experiment/best.pt --output runs/experiment/test_metrics.json --partition test
python -m psdformer predict --data data/windows.npz --checkpoint runs/experiment/best.pt --output runs/experiment/test_predictions.csv --partition test
```

新数据推理使用 `predict --partition all`。测试集评估默认要求与训练时使用的 NPZ 一致，防止无意间用错数据/划分。预测可以输出 `.csv` 或 `.npz`。只导出实际参与训练的任务头；纯分类训练的随机连续头不会冒充已训练预测。

自定义受试者划分时，为 train 命令增加 `--split-json splits.json`。JSON 包含 `train`、`val`、`test` 三个 ID 列表，互不相交且覆盖全部受试者。每个集合都需非空。不要用测试指标选择训练超参数。

输出包含 `best.pt`、`last.pt`、`config.json`、`split.json`、`history.json`、`training_summary.json` 等文件。独立 evaluate/predict 命令生成相应指标/预测文件。50 轮是示例默认值，不是论文公布值；修改配置的 `training.epochs` 即可。每轮会输出训练和验证损失。

目前未提供断点续训命令。检查点包含优化器和随机状态，但没有声称只加载它就能精确续训。请使用不同输出目录保存不同实验，避免覆盖旧结果。

### CSV 转换示例

```bash
python -m psdformer prepare-csv --input recordings.csv --output data/windows.npz --feature-columns pulse respiration EHS RHS PPS --target-columns stress --subject-column subject --session-column session --time-column timestamp --input-len 24 --pred-len 3 --stride 1
```

上面的列名是示例，必须替换为真实字段。CSV 可以包含多个受试者/会话，窗口按受试者和会话分别生成，不会跨边界。时间戳应为可排序的数值；该入口不负责传感器标定或不规则采样重采样。分类任务另外指定 `--label-column label --num-classes C`；没有真实标签时不要随意填充。仅分类可省略 `--target-columns`。

若将预测跨度更改为 6、12 或 24，需要同时更改数据构造的 `--pred-len` 和模型配置的 `pred_len`。论文 K 的含义不够完整，本实现将其作为未来预测步数，这是明确的实现解释。

本包不包含论文全部对比基线和消融实验，也未实现图9未定义的预训练教师/一致性损失或图12的邻接矩阵重建；这些与正文方法的关系没有交代。实现范围是正文公式及算法对应的完整可运行流程。

实际合成演示输出已附在 `examples/smoke_output`，CSV 格式示例见 `examples/README.md`。这些文件不能代替真实数据验证。
