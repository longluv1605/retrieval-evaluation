from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None


def read_image_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image list: {path}")
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        names.append(ln)
    if not names:
        raise ValueError(f"Empty image list: {path}")
    return names


def parse_timestamp_scale(unit: str) -> float:
    unit_norm = unit.strip().lower()
    scales = {
        "s": 1.0,
        "ms": 1e-3,
        "us": 1e-6,
        "ns": 1e-9,
    }
    if unit_norm not in scales:
        allowed = ", ".join(sorted(scales))
        raise ValueError(f"Invalid timestamp unit: {unit}. Allowed: {allowed}")
    return scales[unit_norm]


def read_timestamps(query_root: Path) -> dict[str, float]:
    path = query_root / "timestamps.txt"
    if not path.exists():
        return {}

    out: dict[str, float] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        ln = line.strip()
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def timestamp_to_human(ts_seconds: float) -> str:
    if not (0.0 <= ts_seconds <= 4102444800.0):
        return "n/a"
    try:
        dt = datetime.fromtimestamp(ts_seconds, tz=timezone.utc)
        return dt.isoformat()
    except Exception:
        return "n/a"


def load_existing_sequences(path: Path) -> list[list[str]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Invalid sequence json at {path}: top-level must be a list")

    out: list[list[str]] = []
    for seq in payload:
        if not isinstance(seq, list):
            continue
        row = [x for x in seq if isinstance(x, str)]
        if row:
            out.append(row)
    return out


def write_sequences(path: Path, sequences: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [seq for seq in sequences if seq]
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


@dataclass
class QueryItem:
    name: str
    raw_ts: float
    ts_seconds: float


class SequenceAnnotatorApp:
    def __init__(
        self,
        query_root: Path,
        output_json: Path,
        timestamp_unit: str,
        dt_s_default: float,
    ) -> None:
        try:
            import tkinter as tk  # pylint: disable=import-outside-toplevel
            from tkinter import messagebox  # pylint: disable=import-outside-toplevel
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Tkinter is not available in this Python environment. "
                "Please install python3-tk (system package) and retry."
            ) from exc

        self.tk = tk
        self.messagebox = messagebox
        self.query_root = query_root
        self.output_json = output_json
        self.timestamp_unit = timestamp_unit
        self.dt_s_default = dt_s_default

        names_path = query_root / "query_names.txt"
        if not names_path.exists():
            names_path = query_root / "images.txt"
        names = read_image_list(names_path)

        scale = parse_timestamp_scale(timestamp_unit)
        ts_raw = read_timestamps(query_root)

        items: list[QueryItem] = []
        for idx, name in enumerate(names):
            raw = ts_raw.get(name)
            if raw is None:
                raw = float(idx) * float(dt_s_default) / scale
            ts_seconds = raw * scale
            items.append(QueryItem(name=name, raw_ts=raw, ts_seconds=ts_seconds))

        self.items_sorted = sorted(items, key=lambda x: (x.ts_seconds, x.name))

        self.completed_sequences = load_existing_sequences(output_json)
        annotated = {n for seq in self.completed_sequences for n in seq}
        self.pending_items = [it for it in self.items_sorted if it.name not in annotated]

        self.current_sequence: list[str] = []
        self.current_index = 0

        self.root = self.tk.Tk()
        self.root.title("Sequence Annotator")
        self.root.geometry("1200x820")

        top = self.tk.Frame(self.root)
        top.pack(fill=self.tk.X, padx=12, pady=8)

        self.progress_label = self.tk.Label(top, text="")
        self.progress_label.pack(anchor="w")

        self.timestamp_label = self.tk.Label(top, text="")
        self.timestamp_label.pack(anchor="w")

        self.path_label = self.tk.Label(top, text="")
        self.path_label.pack(anchor="w")

        self.sequence_label = self.tk.Label(top, text="")
        self.sequence_label.pack(anchor="w")

        img_frame = self.tk.Frame(self.root)
        img_frame.pack(fill=self.tk.BOTH, expand=True, padx=12, pady=8)

        self.image_label = self.tk.Label(img_frame, text="", anchor="center")
        self.image_label.pack(fill=self.tk.BOTH, expand=True)

        controls = self.tk.Frame(self.root)
        controls.pack(fill=self.tk.X, padx=12, pady=10)

        self.btn_add = self.tk.Button(controls, text="Add To Sequence", command=self.on_add)
        self.btn_add.pack(side=self.tk.LEFT, padx=6)

        self.btn_save_seq = self.tk.Button(
            controls,
            text="Save And Create Seq",
            command=self.on_save_and_create_seq,
        )
        self.btn_save_seq.pack(side=self.tk.LEFT, padx=6)

        self.btn_save_only = self.tk.Button(controls, text="Save Progress", command=self.on_save_only)
        self.btn_save_only.pack(side=self.tk.LEFT, padx=6)

        self.root.bind("<a>", lambda _e: self.on_add())
        self.root.bind("<s>", lambda _e: self.on_save_and_create_seq())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self._photo_ref: Any = None
        self.refresh_view()

    def current_item(self) -> QueryItem | None:
        if self.current_index >= len(self.pending_items):
            return None
        return self.pending_items[self.current_index]

    def refresh_view(self) -> None:
        total = len(self.pending_items)
        idx = self.current_index
        item = self.current_item()

        self.progress_label.configure(
            text=(
                f"Pending image: {idx + 1}/{max(total, 1)} | "
                f"Completed sequences: {len(self.completed_sequences)} | "
                f"Current sequence size: {len(self.current_sequence)}"
            )
        )

        if item is None:
            self.timestamp_label.configure(text="No pending images. Save current sequence if needed.")
            self.path_label.configure(text=f"Output: {self.output_json}")
            self.sequence_label.configure(text=f"Current sequence preview: {self.current_sequence[:8]}")
            self.image_label.configure(text="All annotated.", image="")
            self._photo_ref = None
            return

        human = timestamp_to_human(item.ts_seconds)
        self.timestamp_label.configure(
            text=(
                f"Timestamp raw={item.raw_ts} ({self.timestamp_unit}), "
                f"seconds={item.ts_seconds:.6f}, human_utc={human}"
            )
        )
        self.path_label.configure(text=f"Image: {item.name}")
        self.sequence_label.configure(text=f"Current sequence preview: {self.current_sequence[:8]}")

        image_path = self.query_root / "images" / item.name
        if Image is None or ImageTk is None:
            self.image_label.configure(text=f"Pillow not available. Image path: {image_path}", image="")
            self._photo_ref = None
            return

        if not image_path.exists():
            self.image_label.configure(text=f"Image not found: {image_path}", image="")
            self._photo_ref = None
            return

        try:
            with Image.open(image_path) as im:
                im = im.convert("RGB")
                max_w, max_h = 1120, 640
                scale = min(max_w / im.width, max_h / im.height)
                new_w = max(1, int(im.width * scale))
                new_h = max(1, int(im.height * scale))
                im = im.resize((new_w, new_h))
                photo = ImageTk.PhotoImage(im)
            self._photo_ref = photo
            self.image_label.configure(image=photo, text="")
        except Exception as exc:
            self.image_label.configure(text=f"Cannot load image: {image_path}\n{exc}", image="")
            self._photo_ref = None

    def on_add(self) -> None:
        item = self.current_item()
        if item is None:
            return
        if item.name not in self.current_sequence:
            self.current_sequence.append(item.name)
        self.current_index += 1
        self.refresh_view()

    def on_save_and_create_seq(self) -> None:
        if self.current_sequence:
            self.completed_sequences.append(list(self.current_sequence))
            self.current_sequence = []
            write_sequences(self.output_json, self.completed_sequences)
            self.messagebox.showinfo("Saved", f"Saved sequence file: {self.output_json}")
        else:
            write_sequences(self.output_json, self.completed_sequences)
            self.messagebox.showinfo("Saved", f"No active sequence. Progress saved to: {self.output_json}")
        self.refresh_view()

    def on_save_only(self) -> None:
        write_sequences(self.output_json, self.completed_sequences + ([self.current_sequence] if self.current_sequence else []))
        self.messagebox.showinfo("Saved", f"Progress saved: {self.output_json}")

    def on_close(self) -> None:
        write_sequences(self.output_json, self.completed_sequences + ([self.current_sequence] if self.current_sequence else []))
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate query image sequences with a Tkinter UI.")
    parser.add_argument("--query-root", type=Path, required=True, help="Path to query set folder.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Output sequence annotation JSON path. Default: <query-root>/query_sequences.json",
    )
    parser.add_argument("--timestamp-unit", type=str, default="us", help="Timestamp unit: s|ms|us|ns")
    parser.add_argument("--dt-s-default", type=float, default=10.0, help="Fallback dt in seconds when timestamp missing")

    args = parser.parse_args()
    output_json = args.output_json or (args.query_root / "query_sequences.json")

    app = SequenceAnnotatorApp(
        query_root=args.query_root,
        output_json=output_json,
        timestamp_unit=args.timestamp_unit,
        dt_s_default=args.dt_s_default,
    )
    app.run()


if __name__ == "__main__":
    main()
