# Docker image for the shell environment for the AI to work in

FROM ubuntu:latest

# Update apt and install nano and vim
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y nano vim

# Set environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# Set the working directory to /app
WORKDIR /app

# Start a shell when the container launches
CMD ["/bin/bash"]
