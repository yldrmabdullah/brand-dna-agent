# 🧬 Brand DNA Agent

**Autonomous Brand Intelligence for the Fashion Industry.**

Brand DNA Agent is a high-performance AI agent that crawls fashion brand websites and social presence to extract structured "Brand DNA" dossiers. It combines advanced computer vision (FashionCLIP), aesthetic clustering, and multimodal LLMs to produce strategic insights in PDF and JSON formats.

![Dashboard Overview](./docs/assets/dashboard.png)

## 🚀 Key Features

- **Autonomous Discovery:** Brand-agnostic crawling using `robots.txt`, sitemaps, and JSON-LD. No custom scrapers needed.
- **Multimodal Intelligence:** 
    - **Vision:** FashionCLIP-based garment classification and quality filtering.
    - **Aesthetics:** K-Means clustering in CLIP space to identify visual style "DNA".
    - **Synthesis:** LLM-driven brand voice extraction and audience profiling.
- **Strategic Outputs:**
    - 📄 **PDF Dossier:** Beautifully rendered, strategist-ready brand books.
    - 🤖 **Machine Manifests:** Refabric-compatible training modules (Look, Mood, Pattern, etc.).
- **Modern Web Dashboard:** A sleek, responsive dark-mode UI to manage brands and monitor runs in real-time.

## 🖼 Interface Preview

| Brand Management | Analysis Report |
|------------------|-----------------|
| ![Brand Detail](./docs/assets/brand_detail.png) | ![Run Report](./docs/assets/run_report.png) |

> *Adding a new brand is as simple as entering a URL:*
> ![New Brand Form](./docs/assets/new_brand.png)

## 📄 Example Outputs

You can find complete examples of the agent's output in the [examples/](./examples) directory:
- [PDF Brand DNA Dossier (Acne Studios)](./examples/acne-studios-dna.pdf)
- [JSON Brand DNA Data](./examples/acne-studios-dna.json)
- [Refabric Training Manifest](./examples/acne-studios-train.json)

## 🛠 Tech Stack

- **Core:** Python 3.11+, Pydantic V2, asyncio
- **Vision/ML:** FashionCLIP, scikit-learn, NumPy
- **LLM Layer:** OpenRouter (Claude 3.5 Sonnet, Gemini 1.5 Pro)
- **Web/API:** FastAPI, Uvicorn, Vanilla JS/CSS (SPA)
- **PDF Engine:** WeasyPrint + Jinja2

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yldrmabdullah/brand-dna-agent.git
cd brand-dna-agent

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
make install-dev
make install-web
```

## 🚦 Quick Start

1. Create a `.env` file with your `OPENROUTER_API_KEY`.
2. Start the web dashboard:
   ```bash
   make serve
   ```
3. Open `http://localhost:8000`, add a brand URL, and click **Run Agent**.

---

## 🏗 Architecture

The agent operates in a 15-stage asynchronous pipeline, managed by a central Orchestrator. It is designed with **graceful degradation**—if one stage fails (e.g., Instagram blocks access), the agent still produces the best possible dossier using remaining signals.

---
*Developed for strategic fashion analysis and AI training preparation.*
