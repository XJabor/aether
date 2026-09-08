"""Launch the RF Baseline Scanner GUI.

    .venv\\Scripts\\python.exe run.py
    .venv\\Scripts\\python.exe run.py --mock
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from PySide6.QtGui import QPalette, QColor
from PySide6.QtWidgets import QApplication

from aether import config


def apply_dark_theme(app: QApplication) -> None:
    """Dark palette, so the spectrum plot is not framed by a white window."""
    app.setStyle("Fusion")
    p = QPalette()
    bg, base, text = QColor("#181d23"), QColor("#12171c"), QColor("#c8d0d8")
    p.setColor(QPalette.Window, bg)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, bg)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, bg)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.ToolTipBase, base)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Highlight, QColor("#2f6f8f"))
    p.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    p.setColor(QPalette.Disabled, QPalette.Text, QColor("#6a737d"))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#6a737d"))
    app.setPalette(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=config.APP_NAME)
    ap.add_argument("--mock", action="store_true",
                    help="start with the synthetic device selected")
    args = ap.parse_args()

    app = QApplication(sys.argv)
    app.setApplicationName(config.APP_NAME)
    apply_dark_theme(app)

    from aether.ui.main_window import MainWindow

    window = MainWindow()
    if args.mock:
        window.mock_action.setChecked(True)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
