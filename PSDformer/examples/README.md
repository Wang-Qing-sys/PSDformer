# CSV 格式示例

synthetic_sessions.csv 是人为构造的平滑数字序列，只有格式演示用途，不是实测数据，也不是经验证的生理仿真。subject/session/timestamp 定义窗口边界与时间顺序，stress 是人为生成的连续示例目标，不是真实压力标签。

在包根目录运行：

```bash
python -m psdformer prepare-csv --input examples/synthetic_sessions.csv --output runs/csv_example/windows.npz --feature-columns pulse respiration EHS RHS PPS --target-columns stress --subject-column subject --session-column session --time-column timestamp --input-len 24 --pred-len 3 --stride 1
```

转换结果与同名 .json 记录描述数据来源；若训练这个示例，务必将其标为合成实验，不能当作真实压力识别结果。
