# Base image with Python
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy source code
COPY ./app ./app
COPY ./data ./data
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]