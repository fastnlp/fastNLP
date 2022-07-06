from rich.highlighter import Highlighter

__all__ = []
class ColorHighlighter(Highlighter):
    def __init__(self, color='black'):
        self.color = color

    def highlight(self, text):
        text.stylize(self.color)