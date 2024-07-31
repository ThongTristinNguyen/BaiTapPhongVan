@echo off
REM Build Docker image
docker build -t mnist-api .

REM Run Docker container
docker run -d -p 80:80 mnist-api
