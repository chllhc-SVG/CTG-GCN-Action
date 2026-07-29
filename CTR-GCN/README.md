# CTR-GCN 动作识别训练项目

本仓库现在聚焦于一条主线：

1. 用摄像头采集动作视频；
2. 用 MediaPipe 提取人体/手部骨架；
3. 整理成数据集；
4. 使用 CTR-GCN / TCN 训练动作识别模型；
5. 评估并导出模型。

## 推荐主线流程

```text
摄像头采集
  -> MediaPipe 提取 pose / hands 骨架
  -> 保存为数据集
  -> 训练动作识别模型
  -> 评估结果
  -> 导出/部署
```

## 仓库里建议保留的内容

```text
CTR-GCN/
  scripts/
    collect_webcam_mediapipe_dataset.py   # 摄像头采集骨架数据集
    train_webcam_mediapipe_tcn.py        # 骨架序列训练脚本
    mediapipe_to_ntu25.py                # 骨架格式转换
    evaluate_pytorch.py                  # 训练后评估
    export_onnx.py                       # 模型导出

  datasets/
    webcam_mediapipe_pose_hands_idle_waving_clapping/
    webcam_mediapipe_full/

  config/
  feeders/
  graph/
  model/
  torchlight/
  requirements.txt
```

## 建议删除或移出主线的杂糅内容

如果你后期只做“采集骨架 -> 数据集 -> 训练动作识别”，下面这些内容建议删掉或移到单独分支，不要和训练主线混在一起：

```text
scripts/mediapipe_tcn_ws_server.py
scripts/webcam_realtime_mediapipe_tcn.py
scripts/run_mediapipe_tcn_ws_server_idle_waving_clapping.bat
scripts/run_webcam_realtime_mediapipe_tcn.bat
docker-compose.yml
Dockerfile
package.json
requirements-digital-human.txt
.dockerignore
```

## 当前动作标签

当前训练标签是：

```text
idle
waving
clapping
stop
```

如果需要中文映射，可以在使用侧处理：

```text
idle -> 无动作
waving -> 挥手
clapping -> 拍手
stop -> 停止
```

## 数据采集建议

建议采集时统一：

- 摄像头分辨率
- 拍摄距离
- 光照环境
- 动作时长
- 录制节奏

这样更利于训练稳定模型。

## 训练建议

建议按下面顺序进行：

1. 运行 `collect_webcam_mediapipe_dataset.py` 采集骨架数据；
2. 整理生成数据集；
3. 用 `train_webcam_mediapipe_tcn.py` 训练；
4. 用 `evaluate_pytorch.py` 查看效果；
5. 必要时再导出 ONNX。

## 本地环境

如果你只做采集和训练，不需要 Docker、不需要 pnpm、不需要 WebSocket 后端。

进入项目目录：

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
```

安装训练依赖：

```powershell
python -m pip install -r requirements.txt
```

## 训练输入和输出

建议将采集出来的数据集、标签文件、训练输出统一放到对应目录中，保持：

- 数据集目录清晰
- 标签文件固定
- 训练结果可复现
- 模型权重单独管理

## 原始 CTR-GCN 训练入口

如果你要继续使用原始 CTR-GCN 的 NTU/UCLA 数据训练，下面这些模块仍然保留：

```text
main.py
ensemble.py
config/
feeders/
graph/
model/
torchlight/
requirements.txt
```

## 一句话建议

如果后期目标是“摄像头骨架采集 -> 数据集 -> 模型训练”，就把实时数字人接入和 Docker/WebSocket 相关文件单独分出去，不要和训练主线混在一个仓库主流程里。
