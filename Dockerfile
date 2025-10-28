# Use an official lightweight Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# We use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
# This includes app.py, the trained model, templates, and params.yaml
COPY app.py .
COPY models/model.pkl ./models/
COPY templates/index.html ./templates/
COPY params.yaml .

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define the command to run the application
# We use gunicorn for a more production-ready server
# You may need to add 'gunicorn' to your requirements.txt file
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]