# XJTLU-URBX-2025
西交利物浦大学2025智能船比赛Team03图像识别脚本 已获得亚军

URBX-2025 比赛版
- 模型：YOLOv5 v6.2 TensorRT .engine，固定 640x640
- 固定：Team=03，conf=0.65，iou=0.45，min_area_ratio=0.05（5%）
- 输出目录：CompetitionData/<Stage>/Round#/Team03/
    ├─ RAW2Hz/
    ├─ Detection01/
    │   ├─ Crop.jpg   # 裁剪(原生分辨率) + BBox + 文本（锐利字体）
    │   └─ RAW.jpg    # 对应全帧
    └─ logs/targets_manifest.json
