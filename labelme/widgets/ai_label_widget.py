from loguru import logger
from PyQt5 import QtWidgets


class AiLabelWidget(QtWidgets.QWidget):
    def __init__(self, on_submit, parent=None):
        super().__init__(parent=parent)

        self.setLayout(QtWidgets.QVBoxLayout())
        self.layout().setSpacing(0)  # type: ignore[union-attr]

        # Create the text input prompt widget (no need for NMS params now)
        text_prompt_widget = _TextPromptWidget(on_submit=on_submit, parent=self)
        text_prompt_widget.setMaximumWidth(400)
        self.layout().addWidget(text_prompt_widget)  # type: ignore[union-attr]

    def get_text_prompt(self) -> str:
        if (
            (layout := self.layout()) is None
            or (item := layout.itemAt(0)) is None
            or (widget := item.widget()) is None
        ):
            logger.warning("Cannot get text prompt")
            return ""
        return widget.get_text_prompt()


class _TextPromptWidget(QtWidgets.QWidget):
    def __init__(self, on_submit, parent=None):
        super().__init__(parent=parent)

        self.setLayout(QtWidgets.QHBoxLayout())
        self.layout().setContentsMargins(0, 0, 0, 0)  # type: ignore[union-attr]

        label = QtWidgets.QLabel(self.tr("AI label"))
        self.layout().addWidget(label)  # type: ignore[union-attr]

        # Text input for the AI prompt
        texts_widget = QtWidgets.QLineEdit()
        texts_widget.setPlaceholderText(self.tr("e.g., dog,cat,bird"))
        self.layout().addWidget(texts_widget)  # type: ignore[union-attr]

        # Submit button for the prompt
        submit_button = QtWidgets.QPushButton(text="Submit", parent=self)
        submit_button.clicked.connect(slot=on_submit)
        self.layout().addWidget(submit_button)  # type: ignore[union-attr]

    def get_text_prompt(self) -> str:
        if (
            (layout := self.layout()) is None
            or (item := layout.itemAt(1)) is None
            or (widget := item.widget()) is None
        ):
            logger.warning("Cannot get text prompt")
            return ""
        return widget.text()
