"""Web API for Brand DNA Agent.

Provides REST endpoints for brand management, run execution, and result viewing.
Serves the web UI as static files.
"""

from __future__ import annotations

import asyncio
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from brand_dna.core.config import BrandConfig, load_brand_config, settings

BRANDS_DIR = Path("configs/brands")
OUTPUTS_DIR = settings.output_dir

# ─── Request / Response models ────────────────────────────────────────────

class BrandCreateRequest(BaseModel):
    name: str
    url: str
    social: dict[str, str] = Field(default_factory=dict)
    known_categories: list[str] = Field(default_factory=list)
    seed_pages: list[str] = Field(default_factory=list)
    notes: str = ""
    crawl: dict[str, Any] = Field(default_factory=dict)
    filter: dict[str, Any] = Field(default_factory=dict)
    analysis: dict[str, Any] = Field(default_factory=dict)


# ─── In-memory run tracker ────────────────────────────────────────────────

_running: dict[str, dict[str, Any]] = {}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _config_filename(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + ".yaml"


def _list_brands() -> list[dict[str, Any]]:
    brands = []
    if not BRANDS_DIR.exists():
        return brands
    for f in sorted(BRANDS_DIR.glob("*.yaml")):
        if f.name == "default.yaml":
            continue
        try:
            raw = yaml.safe_load(f.read_text()) or {}
            slug = _slug(raw.get("name", f.stem))
            runs = _list_runs_for(slug)
            brands.append({
                "name": raw.get("name", f.stem),
                "slug": slug,
                "url": raw.get("url", ""),
                "social": raw.get("social", {}),
                "config_file": f.name,
                "runs_count": len(runs),
                "last_run": runs[0] if runs else None,
            })
        except Exception:
            continue
    return brands


def _list_runs_for(slug: str) -> list[dict[str, Any]]:
    brand_dir = OUTPUTS_DIR / slug
    if not brand_dir.exists():
        return []
    runs = []
    for d in sorted(brand_dir.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        report_path = d / "run_report.json"
        info: dict[str, Any] = {"run_id": d.name, "slug": slug, "status": "unknown"}
        if report_path.exists():
            try:
                report = json.loads(report_path.read_text())
                info.update({
                    "status": "completed",
                    "started_at": report.get("started_at"),
                    "finished_at": report.get("finished_at"),
                    "pages_crawled": report.get("pages_crawled", 0),
                    "images_after_filter": report.get("images_after_filter", 0),
                    "llm_cost": report.get("llm_usage", {}).get("cost_usd", 0),
                    "stages": report.get("stages", []),
                })
            except Exception:
                info["status"] = "error"
        elif d.name in _running:
            info["status"] = "running"
        runs.append(info)
    return runs


# ─── App factory ──────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(title="Brand DNA Agent", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    web_dir = Path(__file__).parent.parent / "web"

    # ── API routes ────────────────────────────────────────────────────────

    @app.get("/api/status")
    async def status():
        has_key = bool(settings.openrouter_api_key)
        return {
            "api_key_set": has_key,
            "output_dir": str(OUTPUTS_DIR),
            "brands_count": len(_list_brands()),
            "models": {
                "primary": settings.model_primary,
                "fast": settings.model_fast,
                "synthesis": settings.model_synthesis,
            },
        }

    @app.get("/api/brands")
    async def list_brands():
        return _list_brands()

    @app.get("/api/brands/{slug}")
    async def get_brand(slug: str):
        for b in _list_brands():
            if b["slug"] == slug:
                config_path = BRANDS_DIR / b["config_file"]
                raw = yaml.safe_load(config_path.read_text()) or {}
                return {**b, "config": raw}
        raise HTTPException(404, "Brand not found")

    @app.post("/api/brands")
    async def create_brand(req: BrandCreateRequest):
        filename = _config_filename(req.name)
        path = BRANDS_DIR / filename
        if path.exists():
            raise HTTPException(409, f"Brand config already exists: {filename}")

        config: dict[str, Any] = {
            "name": req.name,
            "url": req.url,
        }
        if req.social:
            config["social"] = req.social
        if req.known_categories:
            config["known_categories"] = req.known_categories
        if req.seed_pages:
            config["seed_pages"] = req.seed_pages
        if req.notes:
            config["notes"] = req.notes
        if req.crawl:
            config["crawl"] = req.crawl
        if req.filter:
            config["filter"] = req.filter
        if req.analysis:
            config["analysis"] = req.analysis

        # Validate
        try:
            BrandConfig(**config)
        except Exception as exc:
            raise HTTPException(422, f"Invalid config: {exc}")

        BRANDS_DIR.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
        return {"ok": True, "slug": _slug(req.name), "config_file": filename}

    @app.put("/api/brands/{slug}")
    async def update_brand(slug: str, req: BrandCreateRequest):
        for b in _list_brands():
            if b["slug"] == slug:
                path = BRANDS_DIR / b["config_file"]
                config: dict[str, Any] = {"name": req.name, "url": req.url}
                if req.social:
                    config["social"] = req.social
                if req.known_categories:
                    config["known_categories"] = req.known_categories
                if req.seed_pages:
                    config["seed_pages"] = req.seed_pages
                if req.notes:
                    config["notes"] = req.notes
                if req.crawl:
                    config["crawl"] = req.crawl
                if req.filter:
                    config["filter"] = req.filter
                if req.analysis:
                    config["analysis"] = req.analysis
                try:
                    BrandConfig(**config)
                except Exception as exc:
                    raise HTTPException(422, f"Invalid config: {exc}")
                path.write_text(yaml.dump(config, default_flow_style=False, allow_unicode=True))
                return {"ok": True}
        raise HTTPException(404, "Brand not found")

    @app.delete("/api/brands/{slug}")
    async def delete_brand(slug: str):
        for b in _list_brands():
            if b["slug"] == slug:
                path = BRANDS_DIR / b["config_file"]
                path.unlink(missing_ok=True)
                return {"ok": True}
        raise HTTPException(404, "Brand not found")

    @app.post("/api/brands/{slug}/run")
    async def start_run(slug: str):
        brand = None
        for b in _list_brands():
            if b["slug"] == slug:
                brand = b
                break
        if not brand:
            raise HTTPException(404, "Brand not found")

        task_key = f"{slug}"
        if task_key in _running and not _running[task_key].get("done"):
            raise HTTPException(409, "A run is already in progress for this brand")

        config_path = BRANDS_DIR / brand["config_file"]
        brand_config = load_brand_config(config_path)

        async def _do_run():
            try:
                _running[task_key] = {"done": False, "started": datetime.now(timezone.utc).isoformat()}
                orch = Orchestrator(brand_config)
                await orch.run()
                _running[task_key]["done"] = True
                _running[task_key]["status"] = "completed"
            except Exception as exc:
                _running[task_key]["done"] = True
                _running[task_key]["status"] = "failed"
                _running[task_key]["error"] = str(exc)

        asyncio.create_task(_do_run())
        return {"ok": True, "message": "Run started"}

    @app.get("/api/brands/{slug}/runs")
    async def list_runs(slug: str):
        return _list_runs_for(slug)

    @app.get("/api/brands/{slug}/runs/{run_id}")
    async def get_run(slug: str, run_id: str):
        run_dir = OUTPUTS_DIR / slug / run_id
        if not run_dir.exists():
            raise HTTPException(404, "Run not found")
        result: dict[str, Any] = {"run_id": run_id, "slug": slug}
        report = run_dir / "run_report.json"
        if report.exists():
            result["report"] = json.loads(report.read_text())
        dossier = run_dir / "brand_dna.json"
        if dossier.exists():
            result["dossier"] = json.loads(dossier.read_text())
        train = run_dir / "train_modules.json"
        if train.exists():
            result["train_modules"] = json.loads(train.read_text())
        result["has_pdf"] = (run_dir / "brand_dna.pdf").exists()
        return result

    @app.get("/api/brands/{slug}/runs/{run_id}/pdf")
    async def get_pdf(slug: str, run_id: str):
        pdf_path = OUTPUTS_DIR / slug / run_id / "brand_dna.pdf"
        if not pdf_path.exists():
            raise HTTPException(404, "PDF not found")
        return FileResponse(pdf_path, media_type="application/pdf", filename=f"{slug}_brand_dna.pdf")

    @app.get("/api/running")
    async def get_running():
        return _running

    # ── Static + SPA fallback ─────────────────────────────────────────────

    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

    @app.get("/{path:path}")
    async def spa_fallback(path: str):
        if path.startswith("api/"):
            raise HTTPException(404)
        index = web_dir / "index.html"
        if index.exists():
            return HTMLResponse(index.read_text())
        return HTMLResponse("<h1>Brand DNA Agent</h1><p>Web UI not found.</p>")

    return app
