FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set the working directory to /app
WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# installing nltk dependencies
RUN python -c "import stanza; stanza.download('en')"

# Copy the current directory contents into the container at /app
COPY . .

# Run app.py when the container launches
CMD ["python3"]