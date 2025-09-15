from textual.widgets import Static


class CurvesWidget(Static):
    def __init__(self) -> None:
        super().__init__()
        self.border_title = "_curves"
