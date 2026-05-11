"""HTML → PDF renderer via WeasyPrint.

Why WeasyPrint over ReportLab:
- HTML/CSS is the right authoring medium for a designer-readable document.
  A strategist reviewing this isn't reading a JSON dump dressed up as a PDF
  (the case explicitly calls this out as the failure mode); they're reading
  a magazine-quality layout.
- We get @page rules, page numbering, paginated tables, and proper image
  embedding without bespoke layout code.
- Template lives in `templates/brand_dna.html`, so a designer can iterate on
  the layout without touching Python.

WeasyPrint requires Pango/Cairo/GDK-PixBuf system libraries — these are
pinned in the Dockerfile.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from brand_dna.core.models import BrandDNADossier, ImageRecord
from brand_dna.core.observability import get_logger

logger = get_logger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


class PDFRenderer:
    """Stateless renderer. Owns Jinja environment + WeasyPrint integration."""

    def __init__(self) -> None:
        self.env = Environment(
            loader=FileSystemLoader(_TEMPLATES_DIR),
            autoescape=select_autoescape(["html", "xml"]),
        )

    def render_html(
        self,
        dossier: BrandDNADossier,
        images: list[ImageRecord],
    ) -> str:
        """Returns the populated HTML — useful for debugging and CI snapshot tests."""
        # Resolve image_id → absolute filesystem path so file:// URLs work
        # inside WeasyPrint's renderer.
        image_paths = {
            img.image_id: str(Path(img.local_path).resolve()) for img in images
        }
        template = self.env.get_template("brand_dna.html")
        return template.render(
            dossier=dossier,
            image_paths=image_paths,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )

    def render_pdf(
        self,
        dossier: BrandDNADossier,
        images: list[ImageRecord],
        output_path: Path,
    ) -> Path:
        """Render the dossier to PDF on disk. Returns the output path."""
        # Lazy import — WeasyPrint imports a lot of system C libs; we don't want
        # to pay for it on every CLI invocation that doesn't render.
        from weasyprint import HTML

        html_content = self.render_html(dossier, images)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # base_url enables relative paths; we use absolute file:// URLs above
        # so it's mostly defensive.
        HTML(string=html_content, base_url=str(output_path.parent)).write_pdf(
            target=str(output_path)
        )
        logger.info(
            "pdf.rendered",
            path=str(output_path),
            brand=dossier.brand_name,
            size_kb=output_path.stat().st_size // 1024,
        )
        return output_path
