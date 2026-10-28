from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter


@dataclass
class PDFReport:
    """Parameters for PDF report rendering."""
    report_json: dict[str, Any]
    output_path: str
    width: float = letter[0]
    height: float[1] = letter[1]


class ChartBuilder:
    """Build matplotlib severity charts for reports."""

    @staticmethod
    def severity_chart(severity_counts: dict[str, int]) -> Any:
        """Build a severity distribution chart.

        Returns a matplotlib figure object.
        """
        labels = list(severity_counts.keys())
        values = list(severity_counts.values())

        if not labels:
            labels = ["No incidents"]
            values = [0]

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=["red", "orange", "yellow", "green"])
        ax.set_ylabel("Count")
        ax.set_title("Incident Severity Distribution")
        plt.tight_layout()

        return fig
