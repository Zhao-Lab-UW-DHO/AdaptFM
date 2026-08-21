import csv
from pathlib import Path


def format_cell(text: str) -> str:
    """Format URLs as [Link] markdown and trim whitespace."""
    text = text.strip()
    if text.startswith("http://") or text.startswith("https://"):
        return f"[Link]({text})"
    return text


def generate_markdown_table(csv_path: Path) -> str:
    with Path(csv_path).open() as f:
        reader = csv.reader(f)
        rows = list(reader)

        if not rows:
            return ""

        # Columns to exclude (0-indexed: 6 = Code License, 7 = Model License)
        exclude_indices = {6, 7}

        markdown_lines = []

        # Process Header
        header = rows[0]
        filtered_header = [
            col.strip() for i, col in enumerate(header) if i not in exclude_indices
        ]
        markdown_lines.append("| " + " | ".join(filtered_header) + " |")
        markdown_lines.append("| " + " | ".join(["---"] * len(filtered_header)) + " |")

        # Process Rows
        for row in rows[1:]:
            filtered_row = [
                format_cell(col)
                for i, col in enumerate(row)
                if i not in exclude_indices
            ]
            markdown_lines.append("| " + " | ".join(filtered_row) + " |")

        return "\n".join(markdown_lines)


if __name__ == "__main__":
    csv_path = Path(__file__).parent / "model-info.csv"
    markdown_output = generate_markdown_table(csv_path)
    print(markdown_output)
