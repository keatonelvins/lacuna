from textual.widgets import Static


class IOWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.border_title = "_io"
