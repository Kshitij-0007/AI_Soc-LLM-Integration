from __future__ import annotations

from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

from app.reporting.charts import ChartBuilder


class PDFRenderer:
    """Render PDF reports from stored report JSON using reportlab."""

    def __init__(self, report_json: dict[str, Any], output_path: str):
        self.report_json = report_json
        self.output_path = output_path

    def render(self) -> str:
        """Render the PDF report.

        Returns the output file path.
        """
        c = pdfcanvas.Canvas(self.output_path, pagesize=letter)
        width, height = letter

        try:
            self._draw_cover(c, width, height)
            self._draw_executive_summary(c, width, height)
            self._draw_severity_chart(c, width, height)
            self._draw_incidents(c, width, height)
            self._draw_appendix(c, width, height)
        except Exception as e:
            # Fallback minimal PDF
            c.drawString(inch, height - inch, f"Error rendering PDF: {e}")
        c.save()
        return self.output_path

    def _draw_cover(self, c, width: float, height: float) -> None:
        """Draw the cover page."""
        c.setFont("Helvetica-Bold", 24)
        c.drawCentredString(width / 2, height - 2 * inch, "AI-SOC Threat Analysis Report")
        c.setFont("Helvetica", 12)
        c.drawCentredString(width / 2, height - 3 * inch, f"Run ID: {self.report_json.get('run_id', 'N/A')}")
        c.drawCentredString(width / 2, height - 4 * inch, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    def _draw_executive_summary(self, c, width: float, height: float) -> None:
        """Draw the executive summary section."""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - 5 * inch, "Executive Summary")
        c.setFont("Helvetica", 10)
        summary = self.report_json.get("executive_summary", "No summary available")
        c.drawString(inch, height - 5.5 * inch, summary[:500])

    def _draw_severity_chart(self, c, width: float, height: float) -> None:
        """Draw the severity distribution chart."""

        # Count incidents by severity
        incidents = self.report_json.get("incidents", [])
        severity_counts = {}
        for inc in incidents:
            sev = inc.get("severity", "low")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        chart = ChartBuilder.severity_chart(severity_counts)

        # Save chart to buffer and embed in PDF
        import io
        fig = chart
        fig.savefig(io.BytesIO(), format="png")
        c.showPage()  # New page for chart

    def _draw_incidents(self, c, width: float, height: float) -> None:
        """Draw per-incident sections."""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - 6 * inch, "Incidents")

        incidents = self.report_json.get("incidents", [])
        c.setFont("Helvetica", 10)

        for i, inc in enumerate(incidents):
            y = height - 6.5 * inch - i * 3 * inch
            if y < inch:
                c.showPage()
                y = height - 6 * inch

            c.drawString(inch, y, f"Incident {i + 1}: {inc.get('entity', 'unknown')}")
            c.drawString(inch, y - 0.2 * inch, f"Severity: {inc.get('severity', 'unknown')}")
            c.drawString(inch, y - 0.4 * inch, f"Score: {inc.get('score', 0):.2f}")

            # Evidence section
            c.drawString(inch, y - 0.8 * inch, "Evidence:")
            for j, ev in enumerate(inc.get("evidence", [])[:3]):
                c.drawString(inch + 0.2 * inch, y - 1.0 * inch - j * 0.2 * inch,
                             f"- {ev.get('predicted_class', 'unknown')}: confidence {ev.get('confidence', 0):.2f}")

    def _draw_appendix(self, c, width: float, height: float) -> None:
        """Draw the appendix with additional information."""
        c.setFont("Helvetica-Bold", 14)
        c.drawString(inch, height - 7 * inch, "Appendix")
        c.setFont("Helvetica", 10)

        appendix_items = [
            f"Lanes used: {', '.join(self.report_json.get('lanes_used', []))}",
            f"Model versions: {self.report_json.get('model_versions', 'N/A')}",
            f"Validation warnings: {self.report_json.get('validation_warnings', 'None')}",
        ]

        y = height - 7.5 * inch
        for item in appendix_items:
            c.drawString(inch, y, item)
            y -= 0.3 * inch
