# Use an official lightweight Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (model is optional at build time)
RUN mkdir -p models templates
COPY app.py .
COPY templates/index.html ./templates/
COPY params.yaml .
# Copy the checked-in model files into the image so the app can use them directly
COPY models ./models/

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application
# We use gunicorn for a more production-ready server
# You may need to add 'gunicorn' to your requirements.txt file
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]