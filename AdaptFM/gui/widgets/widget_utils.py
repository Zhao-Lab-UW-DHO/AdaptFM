"""
widget_utils.py
---------------
A file for shared elements to ensure consistent UI styling
and UX.
"""

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------

COLOR_SUCCESS_GREEN = "#4caf50"
COLOR_WARNING_AMBER = "#ff9800"
COLOR_ERROR_RED = "#f44336"
COLOR_INFO_BLUE = "#42a5f5"

COLOR_BACKGROUND_PRIMARY = "#2b2b2b"
COLOR_BACKGROUND_SECONDARY = "#1e1e1e"
COLOR_BACKGROUND_HOVER = "#3c3c3c"
COLOR_BACKGROUND_ACTIVE = "#4a4a4a"
COLOR_BACKGROUND_DISABLED = "#2a2a2a"
COLOR_PROGRESS_BACKGROUND = "#333"

COLOR_BORDER_DEFAULT = "#444"
COLOR_BORDER_LIGHT = "#555"

COLOR_TEXT_PRIMARY = "#e8e8e8"
COLOR_TEXT_SECONDARY = "#d4d4d4"
COLOR_TEXT_MUTED = "#888"
COLOR_TEXT_DISABLED = "#666"
COLOR_TEXT_DISABLED_DARK = "#555"
COLOR_TEXT_BLACK = "#000"
COLOR_TEXT_WHITE = "#fff"
COLOR_TEXT_SECTION_HEADER = "#aaa"

COLOR_BUTTON_DISABLED_GREEN = "#3a5a3a"
COLOR_BUTTON_DISABLED_RED = "#5a2a2a"

# ---------------------------------------------------------------------------
# Dimensions & Layouts
# ---------------------------------------------------------------------------

SIZE_ZERO_PX = "0px"
BORDER_WIDTH_DEFAULT_PX = "1px"
BORDER_RADIUS_DEFAULT_PX = "4px"
BORDER_RADIUS_BUTTON_PX = "5px"

SCROLLBAR_WIDTH_PX = "8px"
SCROLLBAR_MIN_HEIGHT_PX = "20px"

PADDING_INPUT_PX = "2px 6px"
PADDING_COMBO_PX = "4px 8px"
PADDING_BTN_SECONDARY_PX = "4px 10px"
PADDING_BTN_PRIMARY_PX = "6px 16px"

FONT_SIZE_SECTION_HEADER_PX = "10px"
FONT_SIZE_HELP_TEXT_PX = "11px"
FONT_SIZE_BUTTON_PX = "13px"

LETTER_SPACING_HEADER_PX = "0.5px"

LAYOUT_MARGINS = (14, 14, 14, 10)
LAYOUT_SPACING = 8
SPLITTER_SIZES = [400, 260]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------
WINDOW_STYLE = f"""
    QWidget {{
        background-color: {COLOR_BACKGROUND_PRIMARY};
        color: {COLOR_TEXT_PRIMARY};
    }}
    QScrollArea {{
        background-color: {COLOR_BACKGROUND_PRIMARY};
        border: none;
    }}
    QScrollBar:vertical {{
        background: {COLOR_BACKGROUND_PRIMARY};
        width: 8px;
    }}
    QScrollBar::handle:vertical {{
        background: #555;
        border-radius: 4px;
        min-height: 20px;
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}
    QSplitter::handle {{
        background: {COLOR_BACKGROUND_PRIMARY};
        height: 1px;
    }}
    QLabel {{
        background: transparent;
        color: {COLOR_TEXT_PRIMARY};
    }}
    QSpinBox, QDoubleSpinBox {{
        background: {COLOR_BACKGROUND_SECONDARY};
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QLineEdit {{
        background: {COLOR_BACKGROUND_SECONDARY};
        color: #d4d4d4;
        border: 1px solid #555;
        border-radius: 4px;
        padding: 2px 6px;
    }}
    QCheckBox {{
        background: transparent;
        color: {COLOR_TEXT_PRIMARY};
    }}
"""
