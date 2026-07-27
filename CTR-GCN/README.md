# CTR-GCN 数字人实时动作识别项目

本项目当前整理后的核心目标是：**数字人前端使用浏览器摄像头采集实时画面，将画面帧通过 WebSocket 发送到后端；后端使用 MediaPipe 提取人体/手部骨架特征，再调用项目内训练好的 TCN 动作识别模型，返回动作状态给数字人前端。**

当前实时识别链路不依赖 OpenMMLab、MMPose、MMAction、MMCV 等组件。

## 当前识别流程

```text
数字人前端摄像头
  -> 前端将实时画面编码为 JPEG/二进制帧
  -> WebSocket: ws://127.0.0.1:8765
  -> CTR-GCN/scripts/mediapipe_tcn_ws_server.py
  -> MediaPipe 提取 pose + hands 骨架
  -> MediaPipe TCN 动作分类模型
  -> 返回 idle / waving / clapping / stop
```

## 保留的核心结构

```text
CTR-GCN/
  scripts/
    mediapipe_tcn_ws_server.py                  # 数字人前端实时识别 WebSocket 服务，核心入口
    train_webcam_mediapipe_tcn.py               # MediaPipe 骨架序列 TCN 训练脚本，同时提供模型结构
    collect_webcam_mediapipe_dataset.py         # 本机摄像头采集训练数据脚本
    webcam_realtime_mediapipe_tcn.py            # 本机摄像头单独测试模型脚本
    run_mediapipe_tcn_ws_server_idle_waving_clapping.bat
                                                   # Windows 一键启动数字人 WebSocket 服务
    run_collect_webcam_mediapipe_dataset.bat    # Windows 一键采集数据
    run_train_webcam_mediapipe_tcn.bat          # Windows 一键训练模型
    run_webcam_realtime_mediapipe_tcn.bat       # Windows 一键本地摄像头测试

  work_dir/
    webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands/
      config.json                               # 当前数字人模型训练配置
      labels.txt                                # 当前动作标签
      labels.json                               # 当前动作标签 JSON
      best_confusion_matrix.txt                 # 训练评估结果
      best_model.pt                             # 训练好的模型权重，需要单独放入或用 Git LFS 管理

  requirements-digital-human.txt                # 数字人实时识别最小依赖
  requirements.txt                              # 原 CTR-GCN 训练环境依赖，较重，不是实时识别必须项
  .gitignore                                    # 已忽略大数据集、训练输出和大模型权重
```

## 已清理的旧入口

已经删除容易混淆的旧实时 CTR-GCN 摄像头入口：

```text
scripts/webcam_realtime_ctrgcn.py
scripts/run_webcam_realtime_ctrgcn.bat
```

也清理了历史 full/face 方案的一键脚本，避免别人拉代码后误走旧流程。

当前推荐只使用：

```text
scripts/mediapipe_tcn_ws_server.py
```

作为数字人前端接入入口。

## 动作标签

当前整理后的数字人实时识别默认标签为：

```text
idle      # 无动作
waving    # 挥手
clapping  # 拍手
stop      # stop / 停止动作
```

如果前端需要显示中文，建议在前端做映射：

```text
idle -> 无动作
waving -> 挥手
clapping -> 拍手
stop -> 停止
```

## 推荐启动方式：Docker + pnpm

别人拉取远端分支后，推荐直接用 Docker 启动识别后端。这样不需要手动配置 Python 环境。下面命令在 Windows PowerShell、Windows CMD、macOS Terminal 中都可以使用。

### 前置要求

本机需要先安装：

- Docker Desktop
- Node.js
- pnpm

Windows 需要先打开 Docker Desktop，并等待 Docker Engine 运行完成。macOS 同样需要先启动 Docker Desktop。

进入 `CTR-GCN` 目录：

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
```

如果是在别人电脑上，进入他自己拉下来的 `CTR-GCN` 目录即可。例如 macOS/Linux：

```bash
cd /path/to/ctrgcn-action-project/CTR-GCN
```

### 确认模型权重存在

启动前必须保证训练好的模型权重存在：

```text
work_dir/webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands/best_model.pt
```

Windows 文件浏览器中对应路径是：

```text
work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt
```

如果远端分支使用 Git LFS 管理模型，请先执行：

```powershell
git lfs pull
```

### 第一次启动或修改依赖后启动

在 `CTR-GCN` 目录下执行：

```powershell
pnpm docker:dev
```

该命令会执行 `docker compose up --build ctrgcn-action-service`。第一次会下载基础镜像和 Python 依赖，耗时较长；构建成功后会启动 WebSocket 动作识别服务。

启动成功后会看到类似输出：

```text
MediaPipe TCN WebSocket server listening on ws://0.0.0.0:8765
```

宿主机和数字人前端访问地址仍然是：

```text
ws://127.0.0.1:8765
```

此时保持该终端运行，然后再启动数字人前端项目。数字人前端连接成功后，终端会显示：

```text
client connected
```

### 日常快速启动，不重新构建依赖

镜像已经构建成功后，平时只需要执行：

```powershell
pnpm docker:start
```

该命令会执行 `docker compose up ctrgcn-action-service`，不会主动重新构建镜像，也不会主动重新下载 `torch`、`mediapipe`、`opencv-python` 等依赖。

如果想后台启动：

```powershell
pnpm docker:start:detached
```

### 手动构建镜像

只有修改了 `Dockerfile`、`requirements-digital-human.txt` 或需要重新构建镜像时才执行：

```powershell
pnpm docker:build
```

### 停止 Docker 服务

在另一个终端进入 `CTR-GCN` 目录后执行：

```powershell
pnpm docker:down
```

该命令只停止并删除当前 compose 容器和网络，不会删除镜像和构建缓存。

查看日志：

```powershell
pnpm docker:logs
```

查看服务状态：

```powershell
pnpm docker:ps
```

检查 Compose 配置：

```powershell
pnpm docker:config
```

### Docker 缓存注意事项

不要随便清理 Docker 的 Images 或 Build cache。以下操作会导致下次重新下载大量依赖：

```powershell
docker system prune -a
docker builder prune
docker image prune -a
```

Docker Desktop 里如果删除了 Images 或 Build cache，下次也会重新下载依赖。日常关闭服务只使用：

```powershell
pnpm docker:down
```

日常启动只使用：

```powershell
pnpm docker:start
```

## 本地 Python 启动方式

如果不使用 Docker，也可以手动安装 Python 环境。

进入项目目录：

```powershell
cd D:\B\python\xiaoke-project\ctrgcn-action-project\CTR-GCN
```

建议安装数字人实时识别最小依赖：

```powershell
python -m pip install -r requirements-digital-human.txt
```

说明：`requirements.txt` 是原始 CTR-GCN 实验环境依赖，内容较重。数字人实时识别优先使用 `requirements-digital-human.txt`。

## 启动数字人 WebSocket 动作识别服务

确保模型权重存在：

```text
work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt
```

然后启动服务：

```powershell
python scripts/mediapipe_tcn_ws_server.py --weights work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt --host 127.0.0.1 --port 8765
```

如果没有 GPU，或者 CUDA 报错，使用 CPU：

```powershell
python scripts/mediapipe_tcn_ws_server.py --weights work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt --host 127.0.0.1 --port 8765 --cpu
```

也可以直接双击运行：

```text
scripts\run_mediapipe_tcn_ws_server_idle_waving_clapping.bat
```

启动成功后终端会出现类似输出：

```text
Loaded work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt on cuda:0; labels=['idle', 'waving', 'clapping', 'stop']; feature_mode=pose_hands
MediaPipe TCN WebSocket server listening on ws://127.0.0.1:8765
```

此时不要关闭终端，让服务保持运行。

## 前端连接地址

数字人前端需要连接：

```text
ws://127.0.0.1:8765
```

前端连接成功后，后端终端会输出：

```text
client connected
```

后端返回的数据格式大致为：

```json
{
  "type": "action_recognition",
  "backend": "mediapipe_tcn",
  "phase": "running",
  "action": {
    "rawLabel": "waving",
    "state": "waving",
    "confidence": 0.92,
    "source": "mediapipe_tcn"
  },
  "top3": [
    { "label": "waving", "state": "waving", "score": 0.92 }
  ],
  "pose": {
    "ready": true,
    "bufferCurrent": 64,
    "bufferTarget": 64
  }
}
```

前端主要读取：

```text
action.state
action.confidence
```

## 如果缺少 best_model.pt

如果启动时报：

```text
Weights not found
```

说明权重文件没有放到对应目录。需要先训练或从远端/网盘/Release/Git LFS 下载模型权重，并放到：

```text
work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt
```

重新训练命令：

```powershell
python scripts/train_webcam_mediapipe_tcn.py --dataset-dir datasets\webcam_mediapipe_pose_hands_idle_waving_clapping --output-dir work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands --labels idle,waving,clapping,stop --feature-mode pose_hands --window-size 64 --epochs 80 --batch-size 16
```

## 单独本机摄像头测试

如果不启动数字人前端，只想用本机摄像头测试模型：

```powershell
python scripts/webcam_realtime_mediapipe_tcn.py --weights work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt --feature-mode pose_hands
```

CPU 模式：

```powershell
python scripts/webcam_realtime_mediapipe_tcn.py --weights work_dir\webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands\best_model.pt --feature-mode pose_hands --cpu
```

摄像头窗口打开后，按 `q` 或 `ESC` 退出。

## 上传远端分支注意事项

为了让别人拉下来后可以直接跑，需要保证：

1. 代码文件已经提交；
2. `requirements-digital-human.txt` 已提交；
3. `work_dir/.../config.json`、`labels.txt`、`labels.json` 已提交；
4. `best_model.pt` 通过 Git LFS、Release、网盘或其他方式提供；
5. `package.json`、`Dockerfile`、`docker-compose.yml`、`.dockerignore` 已提交；
6. README 中的启动命令和实际模型目录保持一致。

当前已经为数字人动作识别后端配置了 Docker 启动入口。别人拉取分支后，只要模型权重也存在，就可以在 `CTR-GCN` 目录下执行：

```powershell
pnpm docker:dev
```

然后启动数字人前端，让前端连接：

```text
ws://127.0.0.1:8765
```

当前 `.gitignore` 已经放开下面这个权重路径，方便使用 Git LFS 提交当前数字人识别模型：

```text
work_dir/webcam_mediapipe_pose_hands_idle_waving_clapping_stop_tcn_pose_hands/best_model.pt
```

建议使用 Git LFS 管理该模型权重。
