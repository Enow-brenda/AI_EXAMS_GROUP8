# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files
COPY app.py requirements.txt /app/

# Copy the current directory contents into the container at /usr/src/app
COPY requirements.txt requirements.txt


# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY templates/ templates/
COPY image_classification_model/ image_classification_model/


# Run the application when the container starts
CMD ["python", "app.py"]