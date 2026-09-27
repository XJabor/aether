"""Ask an LLM about the current scan."""
from __future__ import annotations

import html

from PySide6.QtCore import QObject, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPlainTextEdit, QPushButton,
    QTextBrowser, QVBoxLayout, QWidget,
)

from .. import config
from ..ai import context as ai_context
from ..ai import keys
from ..ai.provider import LLMProvider, Message, ProviderError, make_provider
from .control_panel import shrinkable

SUGGESTIONS = [
    "What are the strongest signals here, and what are they likely to be?",
    "Anything unusual, unexpected, or worth a closer look?",
    "Which of these could be receiver artifacts rather than real signals?",
    "How does this compare with the loaded baseline?",
    "Is my gain set sensibly for this environment?",
]


class _ChatWorker(QObject):
    """Runs one request off the GUI thread."""

    delta = Signal(str)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, provider: LLMProvider, messages: list[Message],
                 system: str) -> None:
        super().__init__()
        self.provider = provider
        self.messages = messages
        self.system = system

    @Slot()
    def run(self) -> None:
        try:
            self.provider.stream_chat(self.messages, self.system, self.delta.emit)
        except ProviderError as exc:
            self.failed.emit(str(exc))
        except Exception as exc:
            self.failed.emit("Unexpected error: %r" % (exc,))
        finally:
            self.finished.emit()


class ChatPanel(QWidget):
    """Chat about the current spectrum.

    The scan summary is attached to the first question only. Re-sending
    thousands of tokens of table on every follow-up would waste the context
    window without telling the model anything it has not already seen.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = config.Settings.load()
        self._history: list[Message] = []
        self._thread: QThread | None = None
        self._worker: _ChatWorker | None = None
        self._streaming = ""
        self._context_sent = False

        self._data = None
        self._meta = None
        self._reference = None
        self._reference_meta = None
        self._region = None

        self._build()
        self._refresh_provider_state()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)

        top = QHBoxLayout()
        self.provider_box = shrinkable(QComboBox())
        self.provider_box.addItem("Anthropic (Claude)", "anthropic")
        self.provider_box.addItem("OpenAI", "openai")
        i = self.provider_box.findData(self.settings.ai_provider)
        if i >= 0:
            self.provider_box.setCurrentIndex(i)
        self.model_box = shrinkable(QComboBox())
        top.addWidget(QLabel("Provider"))
        top.addWidget(self.provider_box, 1)
        top.addWidget(self.model_box, 1)
        root.addLayout(top)

        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        self.key_status.setMinimumWidth(1)
        self.key_status.setStyleSheet("font-size: 11px; color: #e8b34a;")
        root.addWidget(self.key_status)

        self.transcript = QTextBrowser()
        self.transcript.setMinimumWidth(180)
        self.transcript.setOpenExternalLinks(True)
        self.transcript.setStyleSheet(
            "QTextBrowser { background: #12171c; border: 1px solid #2a323a; }"
        )
        root.addWidget(self.transcript, 1)

        self.scope_box = QCheckBox("Selected band only")
        self.scope_box.setToolTip(
            "Sends only the region highlighted on the plot, in full detail, "
            "instead of a summary of the whole scan."
        )
        root.addWidget(self.scope_box)

        self.suggest_box = shrinkable(QComboBox())
        self.suggest_box.addItem("Suggested questions...")
        for s in SUGGESTIONS:
            self.suggest_box.addItem(s)
        self.suggest_box.currentIndexChanged.connect(self._use_suggestion)
        root.addWidget(self.suggest_box)

        self.input = QPlainTextEdit()
        self.input.setPlaceholderText(
            "Ask about this scan...  (Ctrl+Enter to send)"
        )
        self.input.setMaximumHeight(90)
        root.addWidget(self.input)

        row = QHBoxLayout()
        self.context_label = QLabel("")
        self.context_label.setStyleSheet("font-size: 11px; color: #8a949e;")
        self.send_button = QPushButton("Send")
        self.clear_button = QPushButton("New chat")
        row.addWidget(self.context_label, 1)
        row.addWidget(self.clear_button)
        row.addWidget(self.send_button)
        root.addLayout(row)

        self.send_button.clicked.connect(self.send)
        self.clear_button.clicked.connect(self.reset_conversation)
        self.provider_box.currentIndexChanged.connect(self._on_provider_changed)
        self.model_box.currentIndexChanged.connect(self._on_model_changed)

        self._append_system_note(
            "Ask a question about the current scan. The measurement settings, "
            "noise floor, detected signals and band occupancy are sent with "
            "your first question -- the raw spectrum is far too large to send."
        )

    # -- provider --------------------------------------------------------

    def _on_provider_changed(self) -> None:
        self.settings.ai_provider = self.provider_box.currentData()
        self.settings.save()
        self._refresh_provider_state()

    def _on_model_changed(self) -> None:
        model = self.model_box.currentData()
        if not model:
            return
        if self.provider_box.currentData() == "anthropic":
            self.settings.anthropic_model = model
        else:
            self.settings.openai_model = model
        self.settings.save()

    def _refresh_provider_state(self) -> None:
        name = self.provider_box.currentData()
        try:
            provider = make_provider(name)
        except ProviderError as exc:
            self.key_status.setText(str(exc))
            self.send_button.setEnabled(False)
            return

        self.model_box.blockSignals(True)
        self.model_box.clear()
        for m in provider.models:
            self.model_box.addItem(m, m)
        saved = (self.settings.anthropic_model if name == "anthropic"
                 else self.settings.openai_model)
        j = self.model_box.findData(saved)
        self.model_box.setCurrentIndex(j if j >= 0 else 0)
        self.model_box.blockSignals(False)

        if provider.has_key:
            self.key_status.setText("")
            self.key_status.setVisible(False)
            self.send_button.setEnabled(True)
        else:
            self.key_status.setVisible(True)
            self.key_status.setText(
                keys.storage_problem()
                or "No API key saved for %s. Add one under Settings > AI. "
                "Keys are stored in %s, not in any project file."
                % (provider.display_name, keys.STORE_DESCRIPTION)
            )
            self.send_button.setEnabled(False)

    def current_provider(self) -> LLMProvider:
        name = self.provider_box.currentData()
        return make_provider(name, self.model_box.currentData() or "")

    # -- data ------------------------------------------------------------

    def set_scan(self, data, meta=None) -> None:
        self._data, self._meta = data, meta
        self._context_sent = False
        self._update_context_label()

    def set_reference(self, data, meta=None) -> None:
        self._reference, self._reference_meta = data, meta
        self._context_sent = False
        self._update_context_label()

    def set_region(self, lo_hz: float | None, hi_hz: float | None) -> None:
        self._region = None if lo_hz is None else (lo_hz, hi_hz)
        self._update_context_label()

    def _build_context(self) -> str:
        region = self._region if self.scope_box.isChecked() else None
        return ai_context.build_context(
            self._data, self._meta, self._reference, self._reference_meta,
            freq_range=region, settings=self.settings,
        )

    def _update_context_label(self) -> None:
        if self._data is None:
            self.context_label.setText("No scan loaded")
            return
        try:
            n = ai_context.estimate_tokens(self._build_context())
        except Exception:
            self.context_label.setText("")
            return
        self.context_label.setText("scan summary ~%s tokens" % "{:,}".format(n))

    # -- conversation ----------------------------------------------------

    def reset_conversation(self) -> None:
        self._history.clear()
        self._context_sent = False
        self.transcript.clear()
        self._append_system_note("New conversation. The scan summary will be "
                                 "sent again with your next question.")

    def _use_suggestion(self, index: int) -> None:
        if index > 0:
            self.input.setPlainText(SUGGESTIONS[index - 1])
            self.suggest_box.setCurrentIndex(0)
            self.input.setFocus()

    def keyPressEvent(self, event) -> None:
        if (event.key() in (Qt.Key_Return, Qt.Key_Enter)
                and event.modifiers() & Qt.ControlModifier):
            self.send()
            return
        super().keyPressEvent(event)

    @Slot()
    def send(self) -> None:
        if self._thread is not None:
            return
        question = self.input.toPlainText().strip()
        if not question:
            return
        if self._data is None:
            self._append_system_note("Run or load a scan first.")
            return

        try:
            provider = self.current_provider()
        except ProviderError as exc:
            self._append_error(str(exc))
            return

        if self._context_sent:
            payload = question
        else:
            payload = "%s\n\n---\nMy question: %s" % (self._build_context(), question)
            self._context_sent = True

        self._history.append(Message("user", payload))
        self.input.clear()
        self._append_user(question)
        self._begin_assistant()

        worker = _ChatWorker(provider, list(self._history),
                             ai_context.SYSTEM_PROMPT)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.delta.connect(self._on_delta)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(self._on_finished)
        self._worker, self._thread = worker, thread

        self.send_button.setEnabled(False)
        self.send_button.setText("Thinking...")
        thread.start()

    @Slot(str)
    def _on_delta(self, text: str) -> None:
        self._streaming += text
        self._render_streaming()

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self._append_error(message)
        # Drop the unanswered turn so the next question is not paired with a
        # user message that has no reply.
        if self._history and self._history[-1].role == "user":
            self._history.pop()
            self._context_sent = False

    @Slot()
    def _on_finished(self) -> None:
        if self._streaming:
            self._history.append(Message("assistant", self._streaming))
        self._streaming = ""
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(3000)
        self._thread = None
        self._worker = None
        self.send_button.setEnabled(True)
        self.send_button.setText("Send")

    # -- rendering -------------------------------------------------------

    def _append_user(self, text: str) -> None:
        self.transcript.append(
            "<p style='margin-top:12px;'><b style='color:#4ec9f0;'>You</b><br>"
            "%s</p>" % html.escape(text).replace("\n", "<br>")
        )

    def _begin_assistant(self) -> None:
        self._stream_anchor = self.transcript.toHtml()
        self.transcript.append(
            "<p style='margin-top:12px;'><b style='color:#7ee081;'>Assistant</b></p>"
        )

    def _render_streaming(self) -> None:
        # Re-rendering the whole transcript each delta would be O(n^2); this
        # replaces only the streaming paragraph.
        self.transcript.setHtml(
            self._stream_anchor
            + "<p style='margin-top:12px;'><b style='color:#7ee081;'>Assistant</b><br>"
            + html.escape(self._streaming).replace("\n", "<br>")
            + "</p>"
        )
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _append_system_note(self, text: str) -> None:
        self.transcript.append(
            "<p style='color:#8a949e; font-size:11px; margin-top:8px;'>%s</p>"
            % html.escape(text)
        )

    def _append_error(self, text: str) -> None:
        self.transcript.append(
            "<p style='color:#e06c6c; margin-top:8px;'><b>Error</b><br>%s</p>"
            % html.escape(text).replace("\n", "<br>")
        )
