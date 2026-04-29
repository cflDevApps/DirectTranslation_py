import sys
import logging

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit,
    QGroupBox, QFormLayout, QFrame, QMessageBox,
)
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QFont

from src.config import AppConfig
from src.ui.gpu_utils import get_gpu_stats
from src.ui.pipeline_worker import PipelineWorker

logger = logging.getLogger("directtranslation.ui")

LANGUAGES: dict[str, str] = {
    "pt": "Português",
    "en": "English",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "it": "Italiano",
    "ja": "Japonês",
    "zh": "Chinês",
    "ru": "Russo",
    "ar": "Árabe",
}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._config = AppConfig.from_file("config.yaml")
        self._worker: PipelineWorker | None = None

        # Latências acumuladas para exibição
        self._asr_ms = 0.0
        self._tr_ms = 0.0
        self._tts_ms = 0.0

        self._build_ui()
        self._populate_audio_devices()
        self._start_gpu_timer()

    # ── Construção da UI ─────────────────────────────────────────────────

    def _build_ui(self):
        self.setWindowTitle("DirectTranslation")
        self.setMinimumSize(640, 520)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setSpacing(10)
        root.setContentsMargins(14, 14, 14, 14)

        root.addWidget(self._build_config_group())
        root.addWidget(self._build_text_group("Palestrante", "_speaker_text",
                                               "Aguardando transcrição..."))
        root.addWidget(self._build_text_group("Tradução", "_translation_text",
                                               "Aguardando tradução..."))
        root.addWidget(self._build_metrics_bar())
        root.addLayout(self._build_buttons())

    def _build_config_group(self) -> QGroupBox:
        group = QGroupBox("Configuração")
        layout = QFormLayout(group)

        self._device_combo = QComboBox()
        layout.addRow("Dispositivo de áudio:", self._device_combo)

        lang_row = QHBoxLayout()
        self._src_lang_combo = QComboBox()
        self._tgt_lang_combo = QComboBox()
        for code, name in LANGUAGES.items():
            self._src_lang_combo.addItem(name, code)
            self._tgt_lang_combo.addItem(name, code)

        self._set_combo_data(self._src_lang_combo, self._config.translation.source_language)
        self._set_combo_data(self._tgt_lang_combo, self._config.translation.target_language)

        lang_row.addWidget(self._src_lang_combo)
        lang_row.addWidget(QLabel("→"))
        lang_row.addWidget(self._tgt_lang_combo)
        layout.addRow("Idioma:", lang_row)

        return group

    def _build_text_group(self, title: str, attr: str, placeholder: str) -> QGroupBox:
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setMaximumHeight(110)
        edit.setPlaceholderText(placeholder)
        edit.setFont(QFont("Segoe UI", 11))
        layout.addWidget(edit)
        setattr(self, attr, edit)
        return group

    def _build_metrics_bar(self) -> QFrame:
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(10, 5, 10, 5)

        mono = QFont("Consolas", 9)

        self._gpu_label = QLabel("GPU: —")
        self._gpu_label.setFont(mono)

        self._latency_label = QLabel("ASR: —  |  TR: —  |  TTS: —")
        self._latency_label.setFont(mono)
        self._latency_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        layout.addWidget(self._gpu_label, stretch=1)
        layout.addWidget(self._latency_label, stretch=1)
        return frame

    def _build_buttons(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        layout.addStretch()

        self._start_btn = QPushButton("▶   Iniciar")
        self._stop_btn  = QPushButton("■   Parar")
        self._stop_btn.setEnabled(False)

        for btn in (self._start_btn, self._stop_btn):
            btn.setMinimumWidth(130)
            btn.setMinimumHeight(38)

        self._start_btn.clicked.connect(self._on_start)
        self._stop_btn.clicked.connect(self._on_stop)

        layout.addWidget(self._start_btn)
        layout.addWidget(self._stop_btn)
        return layout

    # ── Preenchimento dinâmico ───────────────────────────────────────────

    def _populate_audio_devices(self):
        import sounddevice as sd
        self._device_combo.addItem("Padrão do sistema", None)
        try:
            default_idx = sd.default.device[0]
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    label = f"[{i}]  {dev['name']}"
                    if i == default_idx:
                        label += "  (padrão)"
                    self._device_combo.addItem(label, i)
        except Exception:
            pass

    def _start_gpu_timer(self):
        self._gpu_timer = QTimer(self)
        self._gpu_timer.timeout.connect(self._refresh_gpu)
        self._gpu_timer.start(2000)
        self._refresh_gpu()

    # ── Slots de controle ────────────────────────────────────────────────

    @Slot()
    def _on_start(self):
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._speaker_text.clear()
        self._translation_text.clear()
        self.statusBar().showMessage("Iniciando pipeline...")

        config = self._config_from_ui()
        self._worker = PipelineWorker(config)
        self._worker.transcription_ready.connect(self._on_transcription)
        self._worker.translation_ready.connect(self._on_translation)
        self._worker.tts_complete.connect(self._on_tts_complete)
        self._worker.pipeline_started.connect(self._on_pipeline_started)
        self._worker.pipeline_stopped.connect(self._on_pipeline_stopped)
        self._worker.error_occurred.connect(self._on_error)
        self._worker.start()

    @Slot()
    def _on_stop(self):
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("Encerrando...")
        if self._worker:
            self._worker.stop()

    # ── Slots de atualização da UI ───────────────────────────────────────

    @Slot(str, float)
    def _on_transcription(self, text: str, asr_ms: float):
        self._speaker_text.setPlainText(text)
        self._asr_ms = asr_ms
        self._update_latency_label()

    @Slot(str, str, float, float)
    def _on_translation(self, original: str, translated: str, asr_ms: float, tr_ms: float):
        self._translation_text.setPlainText(translated)
        self._asr_ms = asr_ms
        self._tr_ms = tr_ms
        self._update_latency_label()

    @Slot(float)
    def _on_tts_complete(self, tts_ms: float):
        self._tts_ms = tts_ms
        self._update_latency_label()

    @Slot()
    def _on_pipeline_started(self):
        self.statusBar().showMessage("Pipeline ativo — fale algo...")

    @Slot()
    def _on_pipeline_stopped(self):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage("Pipeline encerrado.")

    @Slot(str)
    def _on_error(self, message: str):
        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self.statusBar().showMessage(f"Erro: {message}")
        QMessageBox.critical(self, "Erro no Pipeline", message)

    @Slot()
    def _refresh_gpu(self):
        stats = get_gpu_stats()
        if stats["available"]:
            self._gpu_label.setText(
                f"GPU: {stats['name']}  |  VRAM: {stats['used_gb']:.1f} / {stats['total_gb']:.0f} GB"
            )
        else:
            self._gpu_label.setText("GPU: não disponível  (modo CPU)")

    # ── Helpers ──────────────────────────────────────────────────────────

    def _config_from_ui(self) -> AppConfig:
        cfg = AppConfig.from_file("config.yaml")
        cfg.translation.source_language = self._src_lang_combo.currentData()
        cfg.translation.target_language = self._tgt_lang_combo.currentData()
        cfg.audio.device = self._device_combo.currentData()
        return cfg

    @staticmethod
    def _set_combo_data(combo: QComboBox, data: str):
        for i in range(combo.count()):
            if combo.itemData(i) == data:
                combo.setCurrentIndex(i)
                return

    def _update_latency_label(self):
        self._latency_label.setText(
            f"ASR: {self._asr_ms:.0f} ms  |  TR: {self._tr_ms:.0f} ms  |  TTS: {self._tts_ms:.0f} ms"
        )

    # ── Encerramento ─────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._worker:
            self._worker.stop()
        event.accept()
