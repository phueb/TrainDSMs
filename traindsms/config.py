from pathlib import Path


class Dirs:
    src = Path(__file__).parent
    root = src.parent
    summaries = root / 'summaries'
    runs = root / 'runs'
    data_for_analysis = root / 'data_for_analysis'


class Figs:
    dpi = 200
    ax_font_size = 14
    leg_font_size = 12
    title_font_size = 12
    tick_font_size = 8
