#!/usr/bin/env python3
"""
交互式 ROI 标定工具 — 支持多 ROI
- TaskA: 画 1 个 ROI，画完按 Enter 保存
- TaskB: 画 4 个 ROI，画完按 Enter 保存

每个 ROI 画完自动确认，绿色显示。
"""
import os
import sys
import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROI_DIR = os.path.join(PROJECT_ROOT, "roi_reference")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")

TASKS = {
    "taskA": {"rotate": 0, "n_roi": 1,
              "video": os.path.join(PROJECT_ROOT, "dataset/taskA/taskA_data_video.mp4")},
    "taskB": {"rotate": 180, "n_roi": 4,
              "video": os.path.join(PROJECT_ROOT, "dataset/taskB/taskB_data_video.mp4")},
}

COLORS = [(0, 255, 0), (255, 200, 0), (0, 200, 255), (200, 100, 255)]


def extract_frame(name):
    cfg = TASKS[name]
    os.makedirs(ROI_DIR, exist_ok=True)
    cap = cv2.VideoCapture(cfg["video"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = int(60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip)

    best_frame, best_score, best_idx = None, -1, skip
    prev_gray, idx = None, skip
    end = total - int(10 * fps)
    step = max(1, int(fps // 2))

    while idx <= end:
        ret, frame = cap.read()
        if not ret:
            break
        if (idx - skip) % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            lap = cv2.Laplacian(blur, cv2.CV_64F).var()
            motion = float(np.mean(cv2.absdiff(blur, prev_gray))) if prev_gray is not None else 99
            prev_gray = blur
            center = blur[H // 4:3 * H // 4, W // 4:3 * W // 4]
            texture = float(np.var(center))
            if motion < 25 and texture > 50:
                s = lap * texture / (motion + 1)
                if s > best_score:
                    best_score, best_frame, best_idx = s, frame.copy(), idx
        idx += 1
    cap.release()

    if best_frame is None:
        cap = cv2.VideoCapture(cfg["video"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        _, best_frame = cap.read()
        cap.release()

    rot = cfg["rotate"]
    if rot == 180:
        best_frame = cv2.rotate(best_frame, cv2.ROTATE_180)
    elif rot == 90:
        best_frame = cv2.rotate(best_frame, cv2.ROTATE_90_CLOCKWISE)
    elif rot == 270:
        best_frame = cv2.rotate(best_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

    path = os.path.join(ROI_DIR, f"{name}_frame.png")
    cv2.imwrite(path, best_frame)
    print(f"  📸 帧 #{best_idx} ({best_idx / fps:.1f}s) → {path}")
    return path, W, H, rot


def unrotate(x1, y1, x2, y2, W, H, angle):
    if angle == 0:
        return x1, y1, x2, y2
    if angle == 180:
        return W - 1 - x2, H - 1 - y2, W - 1 - x1, H - 1 - y1
    if angle == 90:
        return y1, H - 1 - x2, y2, H - 1 - x1
    if angle == 270:
        return W - 1 - y2, x1, W - 1 - y1, x2
    return x1, y1, x2, y2


def main():
    if len(sys.argv) < 2:
        print("用法: python tools/roi_marker.py taskA | taskB"); sys.exit(1)
    raw = sys.argv[1]
    name = next((k for k in TASKS if k.lower() == raw.lower()), None)
    if not name:
        print(f"❌ 未知: {raw}"); sys.exit(1)

    info = TASKS[name]
    img_path, W, H, rot = extract_frame(name)
    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]
    scale = min(1.0, 1200 / max(iw, ih))
    disp = cv2.resize(img, None, fx=scale, fy=scale)
    dh, dw = disp.shape[:2]

    rois = []          # [(x1,y1,x2,y2) in original]
    rois_disp = []     # [(x1,y1,x2,y2) in display]
    rois_img = []      # [(x1,y1,x2,y2) in marker image]

    drawing = False
    drag_start = None
    drag_end = None
    drag_img = None   # preview in image coords

    def mouse(event, x, y, *args):
        nonlocal drawing, drag_start, drag_end, drag_img
        if len(rois) >= info["n_roi"]:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            drag_end = (x, y)
            if drag_start is None or drag_end is None:
                drag_img = None
                return
            x1, y1 = min(drag_start[0], drag_end[0]), min(drag_start[1], drag_end[1])
            x2, y2 = max(drag_start[0], drag_end[0]), max(drag_start[1], drag_end[1])
            ix1, iy1 = int(x1 / scale), int(y1 / scale)
            ix2, iy2 = int(x2 / scale), int(y2 / scale)
            ix1 = max(0, min(ix1, iw - 1))
            iy1 = max(0, min(iy1, ih - 1))
            ix2 = max(0, min(ix2, iw - 1))
            iy2 = max(0, min(iy2, ih - 1))
            if ix2 - ix1 < 10 or iy2 - iy1 < 10:
                drag_img = None
                return
            drag_img = (ix1, iy1, ix2, iy2)
            # The marker image is rotated for convenience.  Save ROI back in
            # original-video coordinates; the pipeline rotates ROI with frame.
            ox1, oy1, ox2, oy2 = unrotate(ix1, iy1, ix2, iy2, W, H, rot)
            rois.append((ox1, oy1, ox2, oy2))
            rois_img.append((ix1, iy1, ix2, iy2))
            rois_disp.append((x1, y1, x2, y2))
            drag_img = None
            drag_start = None
            drag_end = None
            print(f"  ROI#{len(rois)}: [{ox1}, {oy1}, {ox2 - ox1}, {oy2 - oy1}]"
                  f"  ({len(rois)}/{info['n_roi']})")
            if len(rois) >= info["n_roi"]:
                print(f"  全部 {info['n_roi']} 个 ROI 已标定，按 Enter 保存")

    cv2.namedWindow(name)
    cv2.setMouseCallback(name, mouse)

    print(f"\n{'='*55}")
    print(f"  📐 ROI 标定 — {name}  ({info['n_roi']} 个 ROI)")
    print(f"  尺寸: {W}×{H}  旋转: {rot}°")
    print(f"{'='*55}")
    print(f"  🖱 画完一个矩形 → 自动确认 → 画下一个")
    print(f"  Enter → 保存全部并退出")
    print(f"  R → 撤销上一个")
    print(f"  Q → 退出")
    print(f"{'='*55}\n")

    while True:
        canvas = disp.copy()
        # 已确认的 ROIs
        for i, (x1, y1, x2, y2) in enumerate(rois_disp):
            c = COLORS[i % len(COLORS)]
            cv2.rectangle(canvas, (x1, y1), (x2, y2), c, 3)
            ox1, oy1, ox2, oy2 = rois[i]
            label = f"#{i + 1}  ({ox1},{oy1}) {ox2 - ox1}×{oy2 - oy1}"
            cv2.putText(canvas, label, (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
        # 正在拖拽的
        if drawing and drag_start and drag_end:
            cv2.rectangle(canvas, drag_start, drag_end, (255, 200, 0), 2)
        # 进度
        cv2.putText(canvas, f"ROI: {len(rois)}/{info['n_roi']}",
                    (10, dh - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        cv2.imshow(name, canvas)
        key = cv2.waitKey(10) & 0xFF

        if key == ord('q') or key == 27:
            print("  已退出")
            break
        if key == ord('r') and rois:
            rois.pop()
            rois_disp.pop()
            rois_img.pop()
            print(f"  ↩ 撤销 ROI#{len(rois) + 1}")
        if key == 13:  # Enter
            if len(rois) < 1:
                print("  ⚠️  请至少画一个 ROI")
                continue
            if name == "taskB" and len(rois) < 4:
                print(f"  ⚠️  TaskB 需要 4 个 ROI，当前只有 {len(rois)} 个")
                continue
            # 保存
            lst = [[x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2 in rois]
            with open(CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f)
            if len(lst) == 1:
                cfg["tasks"][name]["fixed_roi"] = lst[0]
                cfg["tasks"][name].pop("rois", None)
            else:
                cfg["tasks"][name]["rois"] = lst
                cfg["tasks"][name].pop("fixed_roi", None)
            cfg["tasks"][name]["rotate"] = rot
            cfg["tasks"][name]["rotate_rois_with_frame"] = rot != 0
            with open(CONFIG_PATH, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"\n  已保存 {len(lst)} 个 ROI 到 config.yaml!")
            for i, r in enumerate(lst):
                print(f"     ROI#{i + 1}: [{r[0]}, {r[1]}, {r[2]}, {r[3]}]")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
