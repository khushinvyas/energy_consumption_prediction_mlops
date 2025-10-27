# Multi-stage Dockerfile for our Flask application
# This approach keeps our final image small by separating build and runtime environments

# Stage 1: Build stage - includes all dependencies and development tools
FROM python:3.9-slim as builder

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies needed for building Python packages
# build-essential is needed for compiling some Python packages
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
# Docker will only rebuild this layer if requirements.txt changes
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir reduces image size by not storing the package cache
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime stage - minimal environment for running the application
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Install only the runtime system dependencies
# gunicorn is our production WSGI server
RUN apt-get update && apt-get install -y gunicorn && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
# Copy everything except .git and other unnecessary files
COPY . .

# Create non-root user for security
# Running as root is a security risk in production
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose the port the app runs on
# EXPOSE is documentation for which ports the container listens on
# It doesn't actually publish the port - that's done with -p when running docker
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Health check to verify the application is running correctly
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# Command to run the application with gunicorn
# We use gunicorn instead of flask run because:
# 1. gunicorn is a production-ready WSGI server
# 2. flask run is a development server not suitable for production
# 3. gunicorn can handle multiple concurrent requests
# The format is: gunicorn [options] module:variable
# -w 4 means use 4 worker processes
# -b 0.0.0.0:5000 means bind to all interfaces on port 5000
# app:app means use the app variable in the app.py module
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]