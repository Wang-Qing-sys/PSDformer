# CSV 格式示例

subject/session/timestamp 定义窗口边界与时间顺序，stress 是压力标签。

在包根目录运行：

```bash
python -m psdformer prepare-csv --input examples/synthetic_sessions.csv --output runs/csv_example/windows.npz --feature-columns pulse respiration EHS RHS PPS --target-columns stress --subject-column subject --session-column session --time-column timestamp --input-len 24 --pred-len 3 --stride 1
```
