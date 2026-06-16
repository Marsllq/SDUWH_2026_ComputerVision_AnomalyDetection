import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Tuple, Any, Optional

# Color Palette
COLORS = {
    "bg": "#0a0e1a",
    "card_bg": "#131a2e",
    "card_border": "#1e2a4a",
    "ok": "#00d4aa",
    "ok_dim": "#003d33",
    "ng": "#ff3366",
    "ng_dim": "#4a001a",
    "accent": "#00bcd4",
    "threshold": "#ffc107",
    "text_primary": "#e8edf5",
    "text_secondary": "#7a8aa8"
}

def create_dashboard_frame(
    full_frame: np.ndarray,
    roi_scores: List[Tuple[int, float, bool]],
    unit_info: Optional[Dict[str, Any]],
    history: List[float],
    config: Dict[str, Any],
    ok_count: int = 0,
    ng_count: int = 0
) -> np.ndarray:
    """
    Create a single dashboard frame.
    
    Args:
        full_frame: np.ndarray (HWC BGR) — current video frame
        roi_scores: list of (roi_idx, score, is_ng) for each ROI
        unit_info: dict with unit_id, status ("OK"|"NG"), mean_score, max_score, frame_count
        history: list of max_score for last N completed units for score chart
        config: dict with threshold, task_name, rois
        ok_count: total OK units
        ng_count: total NG units
    
    Returns:
        np.ndarray (HWC RGB) — dashboard image ready for video writing
    """
    # Convert BGR to RGB for matplotlib
    if len(full_frame.shape) == 3 and full_frame.shape[2] == 3:
        frame_rgb = full_frame[:, :, ::-1]
    else:
        frame_rgb = full_frame

    if not roi_scores:
        roi_scores = [(i, 0.0, False) for i, _ in enumerate(config.get("rois", []))]

    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=COLORS["bg"])
    gs = GridSpec(1, 2, width_ratios=[28, 72], figure=fig)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05, wspace=0.05)

    # --- Left Panel ---
    ax_left = fig.add_subplot(gs[0])
    ax_left.set_facecolor(COLORS["bg"])
    ax_left.axis('off')
    
    # Title
    ax_left.text(0.05, 0.95, "ANOMALY DETECTION", color=COLORS["accent"], fontsize=24, fontweight='bold', transform=ax_left.transAxes)
    task_name = config.get("task_name", "Conveyor Belt Inspection")
    ax_left.text(0.05, 0.92, task_name, color=COLORS["text_secondary"], fontsize=14, transform=ax_left.transAxes)

    # Status Box
    status = unit_info.get("status", "WAIT") if unit_info else "WAIT"
    if status == "OK":
        status_color = COLORS["ok"]
        status_bg = COLORS["ok_dim"]
        status_text = "OK"
    elif status == "NG":
        status_color = COLORS["ng"]
        status_bg = COLORS["ng_dim"]
        status_text = "NG"
    elif status == "MOVING":
        status_color = COLORS["threshold"]
        status_bg = COLORS["card_bg"]
        status_text = "MOVING"
    elif status == "INSPECTING":
        status_color = COLORS["accent"]
        status_bg = COLORS["card_bg"]
        status_text = "INSPECTING"
    else:
        status_color = COLORS["text_secondary"]
        status_bg = COLORS["card_bg"]
        status_text = "WAITING"
    
    ax_left.text(0.05, 0.82, status_text, color=status_color, fontsize=36, fontweight='bold',
                 bbox=dict(facecolor=status_bg, edgecolor=status_color, boxstyle='round,pad=0.5', linewidth=2),
                 transform=ax_left.transAxes)

    # Current Score
    max_score = unit_info.get("max_score", 0.0) if unit_info else 0.0
    threshold = config.get("threshold", 3.0)
    score_color = COLORS["ng"] if max_score > threshold else COLORS["ok"]
    
    ax_left.text(0.05, 0.72, "CURRENT MAX SCORE", color=COLORS["text_secondary"], fontsize=12, transform=ax_left.transAxes)
    ax_left.text(0.05, 0.65, f"{max_score:.2f}", color=score_color, fontsize=48, fontweight='bold', transform=ax_left.transAxes)
    
    # Amber threshold marker line
    ax_left.plot([0.05, 0.95], [0.63, 0.63], color=COLORS["threshold"], linewidth=2, transform=ax_left.transAxes)
    
    # Metrics Grid
    grid_y = 0.56
    dy = 0.045
    metrics = [
        ("Threshold", f"{threshold:.2f}", COLORS["threshold"]),
        ("Unit ID", str(unit_info.get("unit_id", "-")) if unit_info else "-", COLORS["text_primary"]),
        ("Frame Count", str(unit_info.get("frame_count", 0)) if unit_info else "0", COLORS["text_primary"]),
        ("ROI Count", str(len(roi_scores)), COLORS["text_primary"]),
        ("Blur Gate", unit_info.get("blur_gate", "PASS") if unit_info else "-", COLORS["text_primary"]),
        ("Motion Gate", unit_info.get("motion_gate", "PASS") if unit_info else "-", COLORS["text_primary"]),
        ("Motion", f"{unit_info.get('motion', 0.0):.2f}" if unit_info else "-", COLORS["text_primary"]),
        ("Foreground", f"{unit_info.get('foreground_ratio', 0.0):.3f}" if unit_info else "-", COLORS["text_primary"]),
    ]
    
    for i, (label, val, color) in enumerate(metrics):
        y_pos = grid_y - i * dy
        ax_left.text(0.05, y_pos, label, color=COLORS["text_secondary"], fontsize=14, transform=ax_left.transAxes)
        ax_left.text(0.95, y_pos, val, color=color, fontsize=14, fontweight='bold', ha='right', transform=ax_left.transAxes)

    # Counts
    count_y = 0.22
    ax_left.text(0.05, count_y, "TOTAL OK", color=COLORS["text_secondary"], fontsize=12, transform=ax_left.transAxes)
    ax_left.text(0.05, count_y - 0.04, str(ok_count), color=COLORS["ok"], fontsize=28, fontweight='bold', transform=ax_left.transAxes)
    
    ax_left.text(0.5, count_y, "TOTAL NG", color=COLORS["text_secondary"], fontsize=12, transform=ax_left.transAxes)
    ax_left.text(0.5, count_y - 0.04, str(ng_count), color=COLORS["ng"], fontsize=28, fontweight='bold', transform=ax_left.transAxes)

    # Score History Chart
    if history:
        ax_hist = ax_left.inset_axes([0.05, 0.02, 0.9, 0.12])
        ax_hist.set_facecolor(COLORS["card_bg"])
        ax_hist.tick_params(colors=COLORS["text_secondary"], labelsize=10)
        for spine in ax_hist.spines.values():
            spine.set_color(COLORS["card_border"])
            
        x_vals = list(range(len(history)))
        y_vals = history
        
        # Plot line
        ax_hist.plot(x_vals, y_vals, color=COLORS["text_secondary"], linewidth=1, alpha=0.5, zorder=1)
        
        # Plot points
        colors = [COLORS["ng"] if y > threshold else COLORS["ok"] for y in y_vals]
        ax_hist.scatter(x_vals, y_vals, c=colors, s=16, zorder=2)
        
        # Threshold line
        ax_hist.plot([0, max(1, len(history)-1)], [threshold, threshold], color=COLORS["threshold"], linestyle='--', linewidth=1.5, zorder=3)
        
        ax_hist.set_title("Score History (Last 50 Units)", color=COLORS["text_secondary"], fontsize=10, pad=5)
        ax_hist.set_ylim(0, max(max(y_vals) * 1.2 if y_vals else 0, threshold * 1.5))
        ax_hist.set_xlim(0, max(1, len(history)-1))
        ax_hist.set_xticks([])

    # --- Right Panel ---
    ax_right = fig.add_subplot(gs[1])
    ax_right.axis('off')
    ax_right.imshow(frame_rgb)
    
    # Draw ROIs
    rois = config.get("rois", [])
    is_moving = status in {"MOVING", "WAIT", "WAITING"}
    force_ng = status == "NG"
    force_ok = status == "OK"
    for roi_idx, score, is_ng in roi_scores:
        if roi_idx < len(rois):
            roi = rois[roi_idx]
            x, y, w, h = roi['x'], roi['y'], roi['w'], roi['h']
            if is_moving:
                color = COLORS["threshold"]
            elif force_ng:
                color = COLORS["ng"]
                is_ng = True
            elif force_ok:
                color = COLORS["ok"]
                is_ng = False
            else:
                color = COLORS["ng"] if is_ng else COLORS["ok"]
            
            # Main border
            rect = patches.Rectangle((x, y), w, h, linewidth=3, edgecolor=color, facecolor='none', alpha=0.8)
            ax_right.add_patch(rect)
            
            # Corner accents (L-shapes)
            length = min(w, h) * 0.15
            thickness = 5
            
            # Top-left
            ax_right.plot([x, x + length], [y, y], color=color, linewidth=thickness)
            ax_right.plot([x, x], [y, y + length], color=color, linewidth=thickness)
            # Top-right
            ax_right.plot([x + w - length, x + w], [y, y], color=color, linewidth=thickness)
            ax_right.plot([x + w, x + w], [y, y + length], color=color, linewidth=thickness)
            # Bottom-left
            ax_right.plot([x, x + length], [y + h, y + h], color=color, linewidth=thickness)
            ax_right.plot([x, x], [y + h - length, y + h], color=color, linewidth=thickness)
            # Bottom-right
            ax_right.plot([x + w - length, x + w], [y + h, y + h], color=color, linewidth=thickness)
            ax_right.plot([x + w, x + w], [y + h - length, y + h], color=color, linewidth=thickness)
            
            # Status badge
            if not is_moving:
                badge_text = "NG" if is_ng else "OK"
                badge_bg = COLORS["ng_dim"] if is_ng else COLORS["ok_dim"]
                ax_right.text(x + w + 5, y + 15, badge_text, color=color, fontsize=10, fontweight='bold',
                              bbox=dict(facecolor=badge_bg, edgecolor=color, alpha=0.7, boxstyle='round,pad=0.3'))
            
    # Bottom-right overlay
    if unit_info:
        overlay_text = f"Unit: {unit_info.get('unit_id', '-')} | Frame: {unit_info.get('frame_count', 0)} | Score: {max_score:.2f}"
        ax_right.text(0.98, 0.02, overlay_text, color=COLORS["text_primary"], fontsize=14,
                      ha='right', va='bottom', transform=ax_right.transAxes,
                      bbox=dict(facecolor=COLORS["bg"], alpha=0.7, edgecolor='none', boxstyle='round,pad=0.5'))

    # Render to numpy array
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    img_rgb = np.frombuffer(buf, dtype=np.uint8)
    target_w = config.get("viz_width", 1920)
    target_h = config.get("viz_height", 1080)
    # Handle potential Retina/HiDPI scaling (buffer may be 2x larger)
    expected = target_h * target_w * 3
    if len(buf) == expected:
        img_rgb = img_rgb.reshape((target_h, target_w, 3))
    else:
        # Auto-detect: find integer scale factor
        ratio = int(round(np.sqrt(len(buf) / expected)))
        h, w = target_h * ratio, target_w * ratio
        img_rgb = img_rgb.reshape((h, w, 3))
    
    plt.close(fig)
    return img_rgb

class DashboardRenderer:
    """
    Manages rendering state and creates dashboard frames.
    
    Usage:
        renderer = DashboardRenderer(config)
        for frame in video_stream:
            frame_with_dash = renderer.render(full_frame, roi_scores, unit_info)
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        config fields used: threshold, viz_width=1920, viz_height=1080, 
        task_name, roi_count, rois
        """
        self.config = config
        self.units_history: List[float] = []  # list of max_scores for completed units
        self.ok_count = 0
        self.ng_count = 0
        self.total_units = 0
        self.last_unit_id = None
        self._last_completed_id = None  # 防止同个 unit 的 is_completed=True 重复计数
    
    def render(self, full_frame: np.ndarray, roi_scores: List[Tuple[int, float, bool]], unit_info: Optional[Dict[str, Any]]) -> np.ndarray:
        """
        Main render method.
        Returns np.ndarray (HWC RGB).
        """
        # Update state — count only on incomplete → completed transition
        if unit_info:
            current_unit_id = unit_info.get("unit_id")
            is_completed = unit_info.get("is_completed", False)

            if is_completed and current_unit_id is not None and current_unit_id != self._last_completed_id:
                # First frame this unit is marked completed — count it
                self._last_completed_id = current_unit_id
                self.units_history.append(unit_info.get("max_score", 0.0))
                if len(self.units_history) > 50:
                    self.units_history.pop(0)

                completed_status = unit_info.get("completed_status", unit_info.get("status", "OK"))
                if completed_status == "OK":
                    self.ok_count += 1
                else:
                    self.ng_count += 1

                self.total_units += 1
                self.last_unit_id = current_unit_id
            elif not is_completed and current_unit_id is not None and current_unit_id != self.last_unit_id:
                # New in-progress unit (not yet completed)
                self.units_history.append(unit_info.get("max_score", 0.0))
                if len(self.units_history) > 50:
                    self.units_history.pop(0)
                self.last_unit_id = current_unit_id
            elif not is_completed and self.units_history:
                # Same in-progress unit — update score
                self.units_history[-1] = unit_info.get("max_score", 0.0)
            # else: same completed unit repeated — do nothing (no double-count)
        
        return create_dashboard_frame(
            full_frame, roi_scores, unit_info,
            self.units_history[-50:], self.config,
            self.ok_count, self.ng_count
        )
