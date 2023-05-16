# Docker image for the shell environment for the AI to work in

FROM ubuntu:latest

# Update apt and install nano and vim
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y nano vim build-essential git openjdk-11-jdk-headless

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Start a shell when the container launches
CMD ["/bin/bash"]
