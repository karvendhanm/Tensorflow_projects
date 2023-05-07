FROM tensorflow/tensorflow:latest-gpu-py3

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Run app.py when the container launches
CMD ["python3"]