# Use an official Python runtime as a parent image
FROM python:3.10-slim
USER root

# Set the working directory in the container
WORKDIR /app
# copy python dependencies file
COPY requirements.txt ./
# install dependencies
RUN pip install --no-cache-dir -r ./requirements.txt 
# install ipython
RUN pip install ipython
