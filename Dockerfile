FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    bash \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Clean Windows carriage returns and make start.sh executable
RUN sed -i 's/\r$//' start.sh && chmod +x start.sh

# Expose API port
EXPOSE 8000

# Run the startup script
CMD ["bash", "start.sh"]
