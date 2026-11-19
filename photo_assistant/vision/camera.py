"""Open the requested camera and verify that it delivers a frame."""
from __future__ import annotations

import time

import cv2


def _named_candidates(device_name: str):
    try:
        from cv2_enumerate_cameras import enumerate_cameras
    except ImportError as exc:
        raise RuntimeError(
            "按名称选择摄像头需要 cv2-enumerate-cameras。请用当前 Python 运行 "
            "python -m pip install -r requirements-client.txt。"
        ) from exc

    name = device_name.strip().casefold()
    if not name:
        raise ValueError("摄像头名称不能为空。")
    candidates = []
    names = set()
    unavailable = []
    for backend, label in ((cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_MSMF, "MSMF")):
        try:
            devices = enumerate_cameras(backend)
        except (RuntimeError, OSError, NotImplementedError, cv2.error):
            unavailable.append(label)
            continue
        names.update(camera.name for camera in devices if camera.name)
        matches = [c for c in devices if c.name and c.name.strip().casefold() == name]
        if len(matches) > 1:
            raise RuntimeError(f"{label} 找到了多个同名摄像头“{device_name}”。请用 --camera-device 指定设备编号。")
        if matches:
            candidates.append((matches[0], label))

    if not candidates:
        found = "、".join(sorted(names)) or "无"
        partial = f"；未能枚举 {'、'.join(unavailable)}" if unavailable else ""
        raise RuntimeError(f"未找到摄像头“{device_name}”。当前枚举到的设备：{found}{partial}。请检查连接和设备名称。")
    for attribute in ("vid", "pid"):
        known = {getattr(c, attribute) for c, _ in candidates if getattr(c, attribute) is not None}
        if len(known) > 1:
            raise RuntimeError(f"同名摄像头“{device_name}”的硬件标识不一致。请用 --camera-device 指定设备编号。")
    # Enumerating an explicit backend returns its own raw index. The same
    # physical camera may have a different index under the other backend.
    return [((c.index, c.backend), f"{c.name} ({label} {c.index})") for c, label in candidates]


def _try_capture(open_args):
    capture = None
    ready = False
    try:
        capture = cv2.VideoCapture(*open_args)
        if capture.isOpened():
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            for attempt in range(3):
                if attempt:
                    time.sleep(0.08)
                ok, frame = capture.read()
                if ok and frame is not None and frame.size:
                    ready = True
                    return capture, frame
    except cv2.error:
        pass
    finally:
        if capture is not None and not ready:
            capture.release()
    return None


def open_camera(device_index: int, device_name: str | None):
    """Return (capture, first BGR frame, label) for the selected device."""
    if device_name is None:
        choices = [((device_index,), f"device {device_index}")]
    else:
        choices = _named_candidates(device_name)
    for attempt in range(2):
        if attempt:
            time.sleep(0.2)
        for open_args, label in choices:
            result = _try_capture(open_args)
            if result is not None:
                capture, frame = result
                return capture, frame, label
    tried = "、".join(label for _, label in choices)
    raise RuntimeError(f"摄像头未能打开或读取首帧（{tried}）。请检查连接、相机权限，并关闭可能占用它的应用后重试。")
