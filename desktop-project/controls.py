"""
GestureOS Controls Module
Python 3.9 compatible
Safe optional imports
"""

import os
import subprocess
import webbrowser
from typing import Optional

# ─────────────────────────────────────────────
# OPTIONAL LIBRARIES (SAFE IMPORTS)
# ─────────────────────────────────────────────

# Windows volume control (pycaw)
try:
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    PYCAW_AVAILABLE = True
except Exception:
    PYCAW_AVAILABLE = False

# Brightness control
try:
    import screen_brightness_control as sbc
    SBC_AVAILABLE = True
except Exception:
    SBC_AVAILABLE = False

# Media keys + screenshot
try:
    import pyautogui
    PYAUTO_AVAILABLE = True
except Exception:
    PYAUTO_AVAILABLE = False


# ─────────────────────────────────────────────
# PINCH CONTROL STATE (Python 3.9 SAFE)
# ─────────────────────────────────────────────

_pinch_baseline = None  # type: Optional[float]
_last_pinch_value = 0.0

PINCH_CONTROLS = ["volume", "brightness"]


def reset_pinch():
    global _pinch_baseline
    _pinch_baseline = None


def update_pinch(value: float, control_type: str):
    """
    value: normalized pinch distance (0.0 - 1.0)
    """
    global _pinch_baseline, _last_pinch_value

    if control_type not in PINCH_CONTROLS:
        return

    if _pinch_baseline is None:
        _pinch_baseline = value
        return

    delta = value - _pinch_baseline
    _last_pinch_value = value

    if abs(delta) < 0.02:
        return

    if control_type == "volume":
        if delta > 0:
            execute_control("volume_up", "5")
        else:
            execute_control("volume_down", "5")

    elif control_type == "brightness":
        if delta > 0:
            execute_control("brightness_up", "5")
        else:
            execute_control("brightness_down", "5")

    _pinch_baseline = value


# ─────────────────────────────────────────────
# MAIN CONTROL EXECUTION
# ─────────────────────────────────────────────

def execute_control(control_type: str, param: Optional[str] = None):
    try:
        # ───────── WEB + APP ─────────
        if control_type == "open_webpage":
            if param:
                webbrowser.open(param)

        elif control_type == "open_app":
            if param:
                subprocess.Popen(param, shell=True)

        # ───────── VOLUME (Windows via pycaw) ─────────
        elif control_type in ["volume_up", "volume_down"]:
            if not PYCAW_AVAILABLE:
                print("[Volume] pycaw not installed.")
                return

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_,
                CLSCTX_ALL,
                None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))

            current = volume.GetMasterVolumeLevelScalar()

            step = float(param) / 100.0 if param else 0.05

            if control_type == "volume_up":
                new_vol = min(1.0, current + step)
            else:
                new_vol = max(0.0, current - step)

            volume.SetMasterVolumeLevelScalar(new_vol, None)

        # ───────── BRIGHTNESS ─────────
        elif control_type in ["brightness_up", "brightness_down"]:
            if not SBC_AVAILABLE:
                print("[Brightness] screen-brightness-control not installed.")
                return

            step = int(param) if param else 5
            current = sbc.get_brightness()[0]

            if control_type == "brightness_up":
                sbc.set_brightness(min(100, current + step))
            else:
                sbc.set_brightness(max(0, current - step))

        # ───────── MUTE ─────────
        elif control_type == "mute":
            if not PYCAW_AVAILABLE:
                print("[Mute] pycaw not installed.")
                return

            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_,
                CLSCTX_ALL,
                None
            )
            volume = cast(interface, POINTER(IAudioEndpointVolume))

            is_muted = volume.GetMute()
            volume.SetMute(not is_muted, None)

        # ───────── MEDIA KEYS ─────────
        elif control_type == "media_play_pause":
            if PYAUTO_AVAILABLE:
                pyautogui.press("playpause")

        elif control_type == "media_next":
            if PYAUTO_AVAILABLE:
                pyautogui.press("nexttrack")

        elif control_type == "media_prev":
            if PYAUTO_AVAILABLE:
                pyautogui.press("prevtrack")

        # ───────── SCREENSHOT ─────────
        elif control_type == "screenshot":
            if PYAUTO_AVAILABLE:
                pyautogui.screenshot("gesture_screenshot.png")

        # ───────── WALLPAPER (Windows only basic example) ─────────
        elif control_type == "change_wallpaper":
            if param and os.path.exists(param):
                if os.name == "nt":
                    import ctypes
                    ctypes.windll.user32.SystemParametersInfoW(20, 0, param, 3)
                else:
                    print("[Wallpaper] Not supported on this OS")

        # ───────── CUSTOM COMMAND ─────────
        elif control_type == "custom_command":
            if param:
                subprocess.Popen(param, shell=True)

        else:
            print(f"[Control] Unknown type: {control_type}")

    except Exception as e:
        print(f"[Control Error] {e}")
