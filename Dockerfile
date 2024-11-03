# syntax=docker/dockerfile:1
FROM debian:bookworm
FROM python:3.10.12

WORKDIR /app

# Copy the current directory contents into the container
COPY . .


RUN apt-get update && \
apt-get install -y build-essential python3-dev python3-distutils && \
    apt-get clean

RUN apt-get install -y ffmpeg
RUN apt-get install -y libsentencepiece-dev
RUN apt install -y cmake

# Install dependencies
# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Install FFmpeg
# RUN python -m spacy download en_core_web_sm

# Run the application
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "1000", "main:app"]
