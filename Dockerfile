# Build stage: use python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (needed for some python packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create outputs directory and ensure it has data
RUN mkdir -p outputs

# Set environment variables
ENV PYTHONPATH="/app/src"
ENV DEMO_MODE="true"
ENV PORT=7860

# Expose the port (Hugging Face uses 7860 by default)
EXPOSE 7860

# Run the application
CMD ["python3", "-m", "uvicorn", "brand_dna.api.app:app", "--host", "0.0.0.0", "--port", "7860"]
