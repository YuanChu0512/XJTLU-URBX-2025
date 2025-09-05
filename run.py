#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, cv2, time, json, hashlib, re, math
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from threading import Thread, Event, Lock
import numpy as np
import torch

# ===== YOLOv5 v6.2 路径 =====
YOLOV5_DIR = os.path.expanduser("~/yolov5")
if YOLOV5_DIR not in sys.path:
    sys.path.insert(0, YOLOV5_DIR)

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import non_max_suppression, scale_coords, check_img_size
from utils.torch_utils import select_device

# ===== 固定参数 =====
TEAM_ID        = "03"
CONF_TH        = 0.65
IOU_TH         = 0.45
MIN_AREA_RATIO = 0.05
ENGINE_SIZE    = 640
RAW_HZ         = 2.0
TZ_CST         = timezone(timedelta(hours=8))

# —— 文本排版（更大更清晰）——
MAX_TEXT_COVER   = 0.55   # 文本最多可覆盖裁剪高度的 55%
FONT_MIN         = 0.72   # 最小字号（HERSHEY_PLAIN 对小字号也清晰）
FONT_INIT_FACTOR = 1.70   # 起始字号更大
MAX_COLUMNS      = 2      # 最多两列
PAD              = 3
COL_PAD          = 8

# —— BBox 绘制 ——（只在裁剪图内部画，不改变分辨率）
BOX_COLOR         = (0, 255, 255)   # 黄色
BOX_INSET_PX      = 1               # 向内收1像素，避免刚好压边看不见
BOX_THICK_MIN     = 2               # 线宽下限
BOX_THICK_RATIO   = 0.015           # 线宽 = ratio * min(w,h)，取整并>=下限
DRAW_CENTER_CROSS = True            # 画中心十字
CROSS_COLOR       = (0, 0, 255)     # 红色
CROSS_HALF        = 6               # 十字半径像素
CROSS_THICK       = 1

# ===== 路径/设备 =====
WEIGHTS         = "/home/jetson/yolov5/best.engine"
DEFAULT_CLASSES = "/home/jetson/yolov5/classes.txt"
COMP_ROOT       = "/home/jetson/CompetitionData"
SOURCE_DEFAULT  = "/dev/video0"
DEVICE_DEFAULT  = "0"

# ===== 工具 =====
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(str(p), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def imsave(path: Path, img, quality_jpg: int = 100) -> bool:
    ext = path.suffix.lower()
    try:
        if ext in (".jpg", ".jpeg"):
            return cv2.imwrite(str(path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality_jpg)])
        elif ext == ".png":
            return cv2.imwrite(str(path), img, [int(cv2.IMWRITE_PNG_COMPRESSION), 3])
        else:
            return cv2.imwrite(str(path), img)
    except Exception:
        return cv2.imwrite(str(path), img)

def save_raw_with_hash(img_bgr, path_img: Path):
    ensure_dir(path_img.parent)
    ok = imsave(path_img, img_bgr, 95)
    if not ok:
        raise RuntimeError(f"写入失败: {path_img}")
    digest = sha256_file(path_img)
    with open(path_img.with_suffix(".hash.txt"), "w", encoding="utf-8") as f:
        f.write(f"SHA256  {path_img.name}  {digest}\n")

# ===== 文本换行/多列布局（HERSHEY_PLAIN + LINE_8，锐利） =====
def _text_size(line: str, font, scale: float, thickness: int):
    (w, h), _ = cv2.getTextSize(line, font, scale, thickness)
    return w, h

def _wrap_by_width(line: str, font, scale: float, thickness: int, max_w: int):
    if not line:
        return [""]
    tokens = re.split(r'(\s+|,)', line)
    out, cur = [], ""
    for tk in tokens:
        cand = cur + tk
        w, _ = _text_size(cand, font, scale, thickness)
        if w <= max_w or not cur:
            cur = cand
        else:
            out.append(cur.rstrip()); cur = tk.lstrip()
    if cur:
        out.append(cur.rstrip())
    final = []
    for seg in out:
        while True:
            w, _ = _text_size(seg, font, scale, thickness)
            if w <= max_w or len(seg) <= 1:
                final.append(seg); break
            est = max(1, int(len(seg) * max_w / (w + 1)))
            final.append(seg[:est]); seg = seg[est:]
    return final

def _layout_multicol(lines, font, scale, thick, w_allowed, h_allowed, max_cols=2):
    _, h0 = _text_size("Ag", font, scale, thick)
    line_h = h0 + PAD
    for cols in range(1, max_cols + 1):
        w_per_col = max(24, int((w_allowed - (cols - 1) * COL_PAD) / cols))
        if w_per_col <= 0: continue
        wrapped = []
        for ln in lines:
            wrapped.extend(_wrap_by_width(ln, font, scale, thick, w_per_col))
        if not wrapped:
            return True, [[]], [0], line_h
        N = len(wrapped)
        lines_per_col = int(math.ceil(N / cols))
        columns, col_widths, ok = [], [], True
        for c in range(cols):
            seg = wrapped[c * lines_per_col:(c + 1) * lines_per_col]
            if (len(seg) * line_h + PAD) > h_allowed: ok = False; break
            maxw = 0
            for s in seg:
                tw, _ = _text_size(s, font, scale, thick); maxw = max(maxw, tw)
            if maxw > w_per_col: ok = False; break
            columns.append(seg); col_widths.append(maxw)
        if ok:
            return True, columns, col_widths, line_h
    return False, None, None, line_h

def draw_info_inplace_sharp(
    img: np.ndarray,
    lines,
    corner: str = "tl",
    max_text_cover: float = MAX_TEXT_COVER,
    font_min: float = FONT_MIN,
    font_init_factor: float = FONT_INIT_FACTOR,
    max_cols: int = MAX_COLUMNS,
    pad: int = PAD
):
    """在裁剪图内部绘制锐利文字（HERSHEY_PLAIN + LINE_8 + 1px八向描边）。"""
    h, w = img.shape[:2]
    if h <= 0 or w <= 0 or not lines:
        return img, False

    font  = cv2.FONT_HERSHEY_PLAIN
    thick = 1
    w_allowed = max(24, w - 2 * pad)
    h_allowed = max(24, int(h * max_text_cover))

    scale = max(font_min, min(2.0, w / 640.0 * font_init_factor))
    truncated = False

    def _layout(sc):
        return _layout_multicol(lines, font, sc, thick, w_allowed, h_allowed, max_cols=max_cols)

    while True:
        ok, columns, colw, line_h = _layout(scale)
        if ok:
            total_w = sum(colw) + (len(colw) - 1) * COL_PAD + 2 * pad
            total_h = max(len(col) * line_h + pad for col in columns) if columns else line_h + pad
            if corner == "tl": x0, y0 = pad, pad
            elif corner == "tr": x0, y0 = max(pad, w - total_w + pad), pad
            elif corner == "bl": x0, y0 = pad, max(pad, h - total_h + pad)
            else: x0, y0 = max(pad, w - total_w + pad), max(pad, h - total_h + pad)

            offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            x = x0
            for ci, col in enumerate(columns):
                y = y0 + int(line_h - pad)
                for t in col:
                    for dx, dy in offsets:
                        cv2.putText(img, t, (x+dx, y+dy), font, scale, (0,0,0), thick, lineType=cv2.LINE_8)
                    cv2.putText(img, t, (x, y), font, scale, (255,255,255), thick, lineType=cv2.LINE_8)
                    y += line_h
                x += (colw[ci] + COL_PAD)
            return img, truncated

        new_scale = scale - 0.06
        if new_scale < font_min - 1e-6:
            truncated = True
            scale = font_min
            ok, columns, colw, line_h = _layout(scale)
            if not ok or not columns:
                return img, True
            offsets = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
            total_w = sum(colw) + (len(colw) - 1) * COL_PAD + 2 * pad
            total_h = max(min(len(col) * line_h + pad, h_allowed) for col in columns)
            x0 = pad if corner in ("tl","bl") else max(pad, w-total_w+pad)
            y0 = pad if corner in ("tl","tr") else max(pad, h-total_h+pad)
            x = x0
            for ci, col in enumerate(columns):
                y = y0 + int(line_h - pad)
                max_lines = max(0, int((h_allowed - pad) / line_h))
                for idx, t in enumerate(col):
                    if idx >= max_lines: truncated = True; break
                    for dx, dy in offsets:
                        cv2.putText(img, t, (x+dx, y+dy), font, scale, (0,0,0), thick, lineType=cv2.LINE_8)
                    cv2.putText(img, t, (x, y), font, scale, (255,255,255), thick, lineType=cv2.LINE_8)
                    y += line_h
                x += (colw[ci] + COL_PAD)
            return img, truncated
        scale = new_scale

# —— 在裁剪图内部绘制 bounding box（矩形 + 中心十字），不改变分辨率 ——
def draw_bbox_on_crop(crop: np.ndarray):
    h, w = crop.shape[:2]
    t = max(BOX_THICK_MIN, int(round(min(h, w) * BOX_THICK_RATIO)))
    t = min(t, max(1, min(h, w)//4))  # 线宽不至于过粗
    x1 = max(0, BOX_INSET_PX); y1 = max(0, BOX_INSET_PX)
    x2 = max(0, w - 1 - BOX_INSET_PX); y2 = max(0, h - 1 - BOX_INSET_PX)
    cv2.rectangle(crop, (x1, y1), (x2, y2), BOX_COLOR, thickness=t, lineType=cv2.LINE_8)
    if DRAW_CENTER_CROSS:
        cx, cy = w // 2, h // 2
        cv2.line(crop, (cx - CROSS_HALF, cy), (cx + CROSS_HALF, cy), CROSS_COLOR, CROSS_THICK, lineType=cv2.LINE_8)
        cv2.line(crop, (cx, cy - CROSS_HALF), (cx, cy + CROSS_HALF), CROSS_COLOR, CROSS_THICK, lineType=cv2.LINE_8)
    return crop

def ask(prompt, default=None, valid=None):
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s: s = str(default) if default is not None else ""
        if valid and s not in valid:
            print(f"只允许：{valid}"); continue
        return s

def is_int_string(s: str) -> bool:
    try: int(s); return True
    except: return False

def open_capture(src):
    if is_int_string(src):
        cap = cv2.VideoCapture(int(src), cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(src))
    else:
        cap = cv2.VideoCapture(src)
    return cap

# ===== 采集与 2Hz 保存 =====
class CaptureThread(Thread):
    def __init__(self, source):
        super().__init__(daemon=True)
        self.cap = open_capture(source)
        self.lock = Lock()
        self.frame_id = 0
        self.im0 = None
        self.stop_ev = Event()
        self.ready = Event()
    def run(self):
        if not self.cap.isOpened(): return
        while not self.stop_ev.is_set():
            ok, frame = self.cap.read()
            if not ok: time.sleep(0.01); continue
            with self.lock:
                self.frame_id += 1
                self.im0 = frame
            self.ready.set()
    def get_latest(self):
        with self.lock:
            fid = self.frame_id
            im0 = None if self.im0 is None else self.im0.copy()
        return fid, im0
    def wait_first(self, timeout=5.0):
        return self.ready.wait(timeout)
    def release(self):
        self.stop_ev.set()
        try: self.cap.release()
        except: pass

class RawSaverThread(Thread):
    def __init__(self, outdir: Path, cap_thread: CaptureThread, hz: float = 2.0):
        super().__init__(daemon=True)
        self.outdir = outdir
        self.cap_thread = cap_thread
        self.period = 1.0 / max(0.1, float(hz))
        self.last_fid = 0
        self.idx = 0
        self.stop_ev = Event()
    def run(self):
        next_t = time.monotonic()
        while not self.stop_ev.is_set():
            now = time.monotonic()
            if now < next_t:
                time.sleep(min(0.02, next_t - now)); continue
            next_t += self.period
            fid, im0 = self.cap_thread.get_latest()
            if im0 is None or fid == self.last_fid: continue
            self.last_fid = fid
            self.idx += 1
            save_raw_with_hash(im0, self.outdir / f"frame_{self.idx:06d}.jpg")
    def stop(self):
        self.stop_ev.set()

# ===== 主流程 =====
def main():
    print("=== URBX-2025 ===")

    if not os.path.isfile(WEIGHTS):
        print(f"[ERR] 找不到模型权重：{WEIGHTS}")
        return

    classes_path = DEFAULT_CLASSES if os.path.isfile(DEFAULT_CLASSES) else ask("classes.txt 路径", DEFAULT_CLASSES)
    comp_root = COMP_ROOT
    source    = SOURCE_DEFAULT
    device_id = DEVICE_DEFAULT

    stage    = ask("比赛阶段(group/knockout/final)", "group", {"group","knockout","final"})
    stage_id = ask("阶段编号(数字，final可填1)", "1")
    round_id = ask("Round编号(数字)", "1")
    dur_sec  = float(ask("检测时长(秒，0=不限)", "0") or 0)

    print(f"\n固定参数：Team=Team{TEAM_ID}，conf={CONF_TH}，iou={IOU_TH}，min_area_ratio={MIN_AREA_RATIO}，engine_size={ENGINE_SIZE}")
    if ask("确认开始？(y/n)", "y", {"y","n"}) != "y":
        print("已取消。"); return

    stage_map = {"group": f"Group{stage_id}", "knockout": f"Knockout{stage_id}", "final": "Final"}
    session_dir = Path(comp_root) / stage_map[stage] / f"Round{round_id}" / f"Team{TEAM_ID}"
    RAW2HZ = session_dir / "RAW2Hz"
    LOGS   = session_dir / "logs"
    ensure_dir(RAW2HZ); ensure_dir(LOGS)
    print(f"[INFO] 输出目录：{session_dir}")

    cap_th = CaptureThread(source); cap_th.start()
    if not cap_th.wait_first(5.0):
        print("[ERR] 采集超时，未获得首帧"); cap_th.release(); return

    device = select_device(device_id)
    model  = DetectMultiBackend(WEIGHTS, device=device, dnn=False, data=None, fp16=True)
    stride = int(model.stride) if isinstance(model.stride, int) else int(max(model.stride))
    imgsz  = check_img_size(ENGINE_SIZE, s=stride)
    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    if os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        names = model.names if hasattr(model, "names") else [str(i) for i in range(1000)]

    saver = RawSaverThread(RAW2HZ, cap_th, RAW_HZ); saver.start()
    t0 = time.monotonic()
    print("[INFO] 开始检测...  (Ctrl+C 可中止)")

    best = defaultdict(lambda: {"conf": -1.0, "dir": None})
    det_seq = {}
    last_dbg = 0.0
    warned_trunc = False

    try:
        with torch.no_grad():
            while True:
                fid, im0 = cap_th.get_latest()
                if im0 is None: time.sleep(0.01); continue
                if dur_sec > 0 and (time.monotonic() - t0) >= dur_sec:
                    print("[INFO] 达到设定时长，结束检测"); break

                im = letterbox(im0, (imgsz, imgsz), stride=stride, auto=False)[0]
                im = im.transpose((2, 0, 1))
                im = np.ascontiguousarray(im)
                im = torch.from_numpy(im).to(device)
                im = im.half() if model.fp16 else im.float()
                im /= 255.0
                if im.ndimension() == 3: im = im.unsqueeze(0)

                pred = model(im, augment=False, visualize=False)
                pred = non_max_suppression(pred, CONF_TH, IOU_TH, max_det=1000)

                raw_cnt = sum(len(d) for d in pred)
                passed_cnt = filtered_area = filtered_multi = filtered_best = 0

                for det in pred:
                    if len(det) == 0: continue
                    det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                    H0, W0 = im0.shape[:2]
                    area_min = MIN_AREA_RATIO * (W0 * H0)

                    centers = [(int((xyxy[0]+xyxy[2])//2), int((xyxy[1]+xyxy[3])//2))
                               for *xyxy, _, _ in det.tolist()]

                    for i, (*xyxy, conf_i, cls_i) in enumerate(det.tolist()):
                        x1, y1, x2, y2 = map(int, xyxy)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(W0-1, x2), min(H0-1, y2)
                        if x2<=x1 or y2<=y1: continue
                        if (x2-x1)*(y2-y1) < area_min: filtered_area += 1; continue
                        only_me = True
                        for j,(cx,cy) in enumerate(centers):
                            if j==i: continue
                            if x1<=cx<=x2 and y1<=cy<=y2: only_me=False; break
                        if not only_me: filtered_multi += 1; continue

                        ci = int(cls_i)
                        cname = names[ci] if isinstance(names,(list,tuple)) and ci<len(names) else str(ci)
                        conf_new = float(conf_i)
                        if conf_new <= best[cname]["conf"]:
                            filtered_best += 1; continue

                        if cname not in det_seq:
                            det_seq[cname] = len(det_seq) + 1
                            seq = det_seq[cname]
                            det_dir = session_dir / f"Detection{seq:02d}"
                            det_dir.mkdir(parents=True, exist_ok=True)
                        else:
                            det_dir = best[cname]["dir"]

                        cx0 = int((x1 + x2) / 2)
                        cy0 = int((y1 + y2) / 2)
                        w0  = int(x2 - x1)
                        h0  = int(y2 - y1)
                        ts  = datetime.now(timezone.utc).astimezone(TZ_CST).strftime("%m-%d %H:%M")
                        lines = [
                            f"c=({cx0},{cy0})  wh=({w0},{h0})",
                            f"cls={cname}  p={conf_new:.3f}",
                            ts
                        ]

                        crop = im0[y1:y2, x1:x2].copy()
                        # 先画 BBox（矩形 + 中心十字），再叠文字；均在裁剪内部，分辨率不变
                        crop = draw_bbox_on_crop(crop)
                        crop_drawn, truncated = draw_info_inplace_sharp(
                            crop, lines, corner="tl", max_text_cover=MAX_TEXT_COVER
                        )
                        if truncated and not warned_trunc:
                            warned_trunc = True
                            print("[WARN] 有极小裁剪，文本已尽量显示（分辨率保持不变）。")

                        imsave(det_dir / "Crop.jpg", crop_drawn, 100)
                        imsave(det_dir / "RAW.jpg",  im0,        95)

                        best[cname]["conf"] = conf_new
                        best[cname]["dir"]  = det_dir
                        passed_cnt += 1

                now = time.monotonic()
                if now - last_dbg > 1.0:
                    last_dbg = now
                    print(f"[DBG] raw={raw_cnt}, pass={passed_cnt}, "
                          f"area_filtered={filtered_area}, multi_filtered={filtered_multi}, best_guarded={filtered_best}")

    except KeyboardInterrupt:
        print("\n[INFO] 用户中断。")
    finally:
        try: saver.stop()
        except: pass
        cap_th.release()
        time.sleep(0.1)

        logs = session_dir / "logs"; ensure_dir(logs)
        manifest = logs / "targets_manifest.json"
        summary = {c: {"conf": v["conf"], "det_dir": str(v["dir"])} for c, v in best.items() if v["dir"] is not None}
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"[OK] 完成。数据在：{session_dir}")
        print(f"[OK] manifest: {manifest}")
        print(f"[OK] RAW2Hz 共 {len(list((session_dir / 'RAW2Hz').glob('frame_*.jpg')))} 张")

if __name__ == "__main__":
    main()
