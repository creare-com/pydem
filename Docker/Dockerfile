# Use an official Python runtime as a parent image
FROM python:3.10-slim
USER root

# Set the working directory in the container
WORKDIR /app

# Install pyDEM and dependencies
RUN pip install --no-cache-dir -r ./requirements.txt 
# Install Ipython
RUN pip install ipython


