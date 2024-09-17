# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that Flask will run on
EXPOSE 8080

# Set environment variables
ENV FLASK_APP=app_flowise.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run the Flask application
CMD flask run -h 0.0.0.0 -p 8080
