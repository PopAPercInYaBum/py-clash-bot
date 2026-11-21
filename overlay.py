#!/usr/bin/env python3
"""
pyclashbot live overlay
=======================
Shows an annotated live view of the emulator with:
  - Bounding boxes around each hand card slot + detected card name + role colour
  - Opponent region highlighted + detected enemy card type
  - Elixir counter
  - Right-side info panel (elixir, enemy type, play count, hand list)

Run from the repo root:
    uv run python overlay.py

Requires the same Python env as pyclashbot (PIL, opencv-python, etc.)
Connects to the same ADB device the bot uses (localhost:6520 by default).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from io import BytesIO

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
    import tkinter as tk
    import tkinter.font as tkfont
except ImportError:
    sys.exit("Install Pillow and tkinter:  pip install Pillow")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ADB_SERIAL = "localhost:6520"
EMU_W, EMU_H = 419, 633          # emulator resolution
DISPLAY_SCALE = 1.6               # how much to enlarge the frame
FPS_TARGET = 1.5                  # overlay refresh rate (screenshots are slow)

# Hand card slot centres (from pyclashbot/bot/coords.py)
HAND_CARD_CENTERS: list[tuple[int, int]] = [
    (142, 561),
    (210, 563),
    (272, 561),
    (341, 563),
]
CARD_BOX_W, CARD_BOX_H = 50, 62  # px around each centre

# Opponent arena region (rows in the screenshot)
OPP_Y0, OPP_Y1 = 50, 280
OPP_X0, OPP_X1 = 0, EMU_W

# Elixir pip pixels (row, col) — same as fight.py's _ELIXIR_COORDS
_ELIXIR_COORDS = [
    (613, 149), (613, 165), (613, 188), (613, 212), (613, 240),
    (613, 262), (613, 287), (613, 314), (613, 339), (613, 364),
]
_ELIXIR_BGR = (244, 137, 240)   # BGR order
_ELIXIR_TOL = 65

# Deployment diff threshold (sum of per-pixel mean-channel diff)
_DEPLOY_THRESHOLD = 300_000
_DEPLOY_COOLDOWN_S = 3.0
_ENEMY_STALE_S = 8.0

# ---------------------------------------------------------------------------
# Role + enemy colours (BGR for OpenCV, RGB for tkinter labels)
# ---------------------------------------------------------------------------
_ROLE_BGR: dict[str, tuple[int, int, int]] = {
    "spell":   (0, 165, 255),
    "defend":  (30, 220, 30),
    "push":    (60,  60, 220),
    "support": (220, 220,  30),
    "unknown": (100, 100, 100),
}
_ENEMY_BGR: dict[str, tuple[int, int, int]] = {
    "bridge_rush": ( 30,  30, 255),
    "tank":        ( 30, 130, 255),
    "spell":       (  0, 165, 255),
    "support":     (100, 180,  50),
    "building":    (180,  60, 180),
    "unknown":     (100, 100, 100),
}
_ROLE_HEX: dict[str, str] = {
    "spell":   "#ffa040",
    "defend":  "#40e040",
    "push":    "#6060ff",
    "support": "#e0e040",
    "unknown": "#888888",
}

# ---------------------------------------------------------------------------
# ADB screenshot
# ---------------------------------------------------------------------------

def _adb_screenshot() -> np.ndarray | None:
    try:
        result = subprocess.run(
            ["adb", "-s", ADB_SERIAL, "exec-out", "screencap", "-p"],
            capture_output=True, timeout=6,
        )
        if result.returncode != 0 or not result.stdout:
            return None
        img = Image.open(BytesIO(result.stdout)).convert("RGB")
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Minimal emulator stub so we can call pyclashbot card detection
# ---------------------------------------------------------------------------

class _FakeEmulator:
    def __init__(self) -> None:
        self._frame: np.ndarray | None = None

    def set(self, frame: np.ndarray) -> None:
        self._frame = frame

    def screenshot(self) -> np.ndarray | None:
        return self._frame


# Try importing pyclashbot card detection
_HAS_DETECTION = False
_identify_hand_cards = None
_get_card_role = None

try:
    sys.path.insert(0, ".")
    from pyclashbot.bot.card_detection import (           # noqa: PLC0415
        identify_hand_cards as _ihc,
        check_which_cards_are_available as _cwca,
        create_default_bridge_iar as _cdbi,
    )
    from pyclashbot.bot.fight import _get_card_role as _gcr  # noqa: PLC0415
    _identify_hand_cards = _ihc
    _check_which_cards_are_available = _cwca
    _create_default_bridge_iar = _cdbi
    _get_card_role = _gcr
    _HAS_DETECTION = True
    print("[overlay] pyclashbot card detection loaded ✓")
except Exception as exc:
    print(f"[overlay] card detection unavailable ({exc}) — name boxes will show '?'")
    _check_which_cards_are_available = None
    _create_default_bridge_iar = None


def _read_elixir(frame: np.ndarray) -> int:
    """Read elixir (0-10) from elixir pip pixels."""
    h, w = frame.shape[:2]
    for level in range(10, 0, -1):
        row, col = _ELIXIR_COORDS[level - 1]
        if row < h and col < w:
            b, g, r = int(frame[row, col, 0]), int(frame[row, col, 1]), int(frame[row, col, 2])
            if (abs(b - _ELIXIR_BGR[0]) < _ELIXIR_TOL and
                    abs(g - _ELIXIR_BGR[1]) < _ELIXIR_TOL and
                    abs(r - _ELIXIR_BGR[2]) < _ELIXIR_TOL):
                return level
    return 0


def _classify_enemy(diff_map: np.ndarray) -> str:
    """Classify what type of card the opponent played from pixel diff centroid."""
    h, w = diff_map.shape
    threshold = float(diff_map.max()) * 0.3
    hot = diff_map > threshold
    if not hot.any():
        return "unknown"
    rows, cols = np.where(hot)
    cy = float(rows.mean()) / h
    cx = float(cols.mean()) / w
    y_spread = float(rows.std()) if len(rows) > 1 else 0.0
    x_spread = float(cols.std()) if len(cols) > 1 else 0.0
    if x_spread > w * 0.25 and y_spread > h * 0.15:
        return "spell"
    if cy > 0.70:
        return "bridge_rush"
    if cy < 0.35:
        return "building" if 0.35 < cx < 0.65 else "support"
    return "tank"


# ---------------------------------------------------------------------------
# Overlay application
# ---------------------------------------------------------------------------

class OverlayApp:
    _PANEL_W = 210
    _BG = "#0d0d1a"
    _PANEL_BG = "#13132a"
    _ACCENT = "#7ec8e3"
    _DIM = "#3a3a5a"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("pyclashbot overlay")
        root.configure(bg=self._BG)
        root.resizable(True, True)

        self._emulator = _FakeEmulator()
        self._baseline: np.ndarray | None = None
        self._last_enemy_type = "unknown"
        self._last_enemy_time = 0.0
        self._cooldown_until = 0.0
        self._hand_names: list[str] = ["?", "?", "?", "?"]
        self._hand_roles: list[str] = ["unknown"] * 4
        self._elixir = 0
        self._enemy_play_count = 0
        self._our_play_count = 0
        self._lock = threading.Lock()
        self._pending: dict = {}
        self._photo: ImageTk.PhotoImage | None = None

        # --- Layout ---
        self.canvas = tk.Canvas(root, bg="#050510", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        panel = tk.Frame(root, bg=self._PANEL_BG, width=self._PANEL_W)
        panel.pack(side=tk.RIGHT, fill=tk.Y)
        panel.pack_propagate(False)

        self._build_panel(panel)
        self._schedule()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_panel(self, panel: tk.Frame) -> None:
        def label(parent, text, fg="#c0c0e0", size=9, bold=False, pady=0):
            weight = "bold" if bold else "normal"
            tk.Label(parent, text=text, bg=self._PANEL_BG, fg=fg,
                     font=("Consolas", size, weight)).pack(pady=pady, anchor="w", padx=12)

        def divider(parent):
            tk.Frame(parent, bg=self._DIM, height=1).pack(fill=tk.X, padx=12, pady=6)

        # Title
        tk.Label(panel, text="pyclashbot", bg=self._PANEL_BG, fg=self._ACCENT,
                 font=("Consolas", 14, "bold")).pack(pady=(14, 0))
        tk.Label(panel, text="live overlay", bg=self._PANEL_BG, fg=self._DIM,
                 font=("Consolas", 8)).pack(pady=(0, 10))

        divider(panel)

        # Stat rows
        self._stat_vars: dict[str, tk.StringVar] = {}
        for key, caption in [("elixir", "ELIXIR"), ("enemy", "ENEMY"), ("opp_plays", "OPP PLAYS")]:
            row = tk.Frame(panel, bg=self._PANEL_BG)
            row.pack(fill=tk.X, padx=12, pady=2)
            tk.Label(row, text=caption, bg=self._PANEL_BG, fg=self._DIM,
                     font=("Consolas", 7), width=10, anchor="w").pack(side=tk.LEFT)
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(row, textvariable=var, bg=self._PANEL_BG, fg="#e0e8ff",
                     font=("Consolas", 11, "bold"), anchor="w").pack(side=tk.LEFT)

        divider(panel)
        label(panel, "HAND CARDS", fg=self._DIM, size=7)

        self._hand_vars: list[tk.StringVar] = []
        self._hand_color_labels: list[tk.Label] = []
        for i in range(4):
            row = tk.Frame(panel, bg=self._PANEL_BG)
            row.pack(fill=tk.X, padx=10, pady=1)
            dot = tk.Label(row, text="●", bg=self._PANEL_BG, fg="#444466",
                           font=("Consolas", 9))
            dot.pack(side=tk.LEFT)
            self._hand_color_labels.append(dot)
            var = tk.StringVar(value=f"[{i}] ?")
            self._hand_vars.append(var)
            tk.Label(row, textvariable=var, bg=self._PANEL_BG, fg="#a0d0a0",
                     font=("Consolas", 8), anchor="w").pack(side=tk.LEFT, fill=tk.X, expand=True)

        divider(panel)
        label(panel, "ENEMY LAST SEEN", fg=self._DIM, size=7)
        self._enemy_detail_var = tk.StringVar(value="—")
        tk.Label(panel, textvariable=self._enemy_detail_var, bg=self._PANEL_BG,
                 fg="#ff7070", font=("Consolas", 10, "bold"), anchor="w",
                 wraplength=180, justify=tk.LEFT).pack(fill=tk.X, padx=12, pady=2)

        divider(panel)
        label(panel, f"ADB  {ADB_SERIAL}", fg=self._DIM, size=7)
        self._status_var = tk.StringVar(value="connecting…")
        tk.Label(panel, textvariable=self._status_var, bg=self._PANEL_BG,
                 fg="#606080", font=("Consolas", 7), anchor="w",
                 wraplength=180).pack(fill=tk.X, padx=12)

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def _schedule(self) -> None:
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(int(1000 / FPS_TARGET), self._schedule)

    def _worker(self) -> None:
        t0 = time.time()
        frame = _adb_screenshot()
        fetch_ms = int((time.time() - t0) * 1000)

        if frame is None:
            self.root.after(0, lambda: self._status_var.set("ADB error — retrying"))
            return

        self._emulator.set(frame)

        # --- Elixir ---
        elixir = _read_elixir(frame)

        # --- Hand card detection ---
        # check_which_cards_are_available must run first — it populates the
        # internal battle_iar cache that identify_hand_cards reads from.
        hand_names: list[str] = []
        hand_roles: list[str] = []
        if _HAS_DETECTION:
            try:
                # Seed the bridge baseline (needed for side-detection internals)
                if _create_default_bridge_iar is not None:
                    _create_default_bridge_iar(self._emulator)
                # Populate battle_iar cache; check_side=False skips ADB side-detect
                if _check_which_cards_are_available is not None:
                    _check_which_cards_are_available(self._emulator, check_side=False)
            except Exception:
                pass
            for slot in range(4):
                try:
                    cid = _identify_hand_cards(self._emulator, slot)
                    role = _get_card_role(cid)
                except Exception:
                    cid, role = "?", "unknown"
                hand_names.append(cid or "?")
                hand_roles.append(role)
        else:
            hand_names = ["?"] * 4
            hand_roles = ["unknown"] * 4

        # --- Opponent deployment detection ---
        now = time.time()
        if self._baseline is None:
            self._baseline = frame.copy()

        enemy_type = self._last_enemy_type
        if now >= self._cooldown_until:
            curr = frame[OPP_Y0:OPP_Y1, OPP_X0:OPP_X1].astype(np.float32)
            base = self._baseline[OPP_Y0:OPP_Y1, OPP_X0:OPP_X1].astype(np.float32)
            diff_map = np.abs(curr - base).mean(axis=2)
            if float(diff_map.sum()) >= _DEPLOY_THRESHOLD:
                enemy_type = _classify_enemy(diff_map)
                self._last_enemy_type = enemy_type
                self._last_enemy_time = now
                self._cooldown_until = now + _DEPLOY_COOLDOWN_S
                self._baseline = frame.copy()
                self._enemy_play_count += 1

        stale = (now - self._last_enemy_time) > _ENEMY_STALE_S

        # --- Annotate frame ---
        display = frame.copy()

        # Elixir text (top-left)
        cv2.putText(display, f"elixir  {elixir}",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (220, 100, 240), 1, cv2.LINE_AA)

        # Hand card boxes
        for i, (cx, cy) in enumerate(HAND_CARD_CENTERS):
            x1, y1 = cx - CARD_BOX_W // 2, cy - 8
            x2, y2 = cx + CARD_BOX_W // 2, cy + CARD_BOX_H - 8
            role = hand_roles[i] if i < len(hand_roles) else "unknown"
            color = _ROLE_BGR.get(role, (100, 100, 100))
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            name = hand_names[i] if i < len(hand_names) else "?"
            short = name[:11]
            # label above box
            cv2.putText(display, short, (x1, y1 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)
            # role tag inside box bottom
            cv2.putText(display, role[:7], (x1 + 2, y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.26, color, 1, cv2.LINE_AA)

        # Opponent region box
        opp_color = (40, 40, 50) if stale else _ENEMY_BGR.get(enemy_type, (100, 100, 100))
        cv2.rectangle(display, (OPP_X0, OPP_Y0), (OPP_X1 - 1, OPP_Y1), opp_color, 2)
        opp_label = f"enemy: {enemy_type}" if not stale else "enemy: watching"
        cv2.putText(display, opp_label, (4, OPP_Y0 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, opp_color, 1, cv2.LINE_AA)

        # Scale up
        sw = int(EMU_W * DISPLAY_SCALE)
        sh = int(EMU_H * DISPLAY_SCALE)
        rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (sw, sh), interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(rgb)

        with self._lock:
            self._pending = {
                "img": img,
                "elixir": elixir,
                "enemy": enemy_type if not stale else "—",
                "opp_plays": self._enemy_play_count,
                "hand_names": hand_names,
                "hand_roles": hand_roles,
                "fetch_ms": fetch_ms,
                "stale": stale,
            }

        self.root.after(0, self._apply)

    def _apply(self) -> None:
        with self._lock:
            p = dict(self._pending)
        if not p:
            return

        # Canvas image
        photo = ImageTk.PhotoImage(p["img"])
        self.canvas.config(width=p["img"].width, height=p["img"].height)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        self._photo = photo

        # Stat panel
        self._stat_vars["elixir"].set(str(p["elixir"]))
        self._stat_vars["enemy"].set(p["enemy"])
        self._stat_vars["opp_plays"].set(str(p["opp_plays"]))

        # Hand slots
        for i, (var, dot) in enumerate(zip(self._hand_vars, self._hand_color_labels)):
            name = p["hand_names"][i] if i < len(p["hand_names"]) else "?"
            role = p["hand_roles"][i] if i < len(p["hand_roles"]) else "unknown"
            var.set(f"[{i}] {name[:14]}")
            dot.config(fg=_ROLE_HEX.get(role, "#444466"))

        # Enemy detail
        enemy = p["enemy"]
        self._enemy_detail_var.set(enemy if enemy != "—" else "none yet")

        # Status bar
        self._status_var.set(
            f"fetch {p['fetch_ms']}ms  |  {'live' if not p['stale'] else 'watching'}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    root = tk.Tk()
    app = OverlayApp(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
