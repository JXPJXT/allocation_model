# Use a stable, production-ready Python image (e.g., 3.11 or 3.12)
FROM python:3.11-slim

# Set environment variables for the application
ENV PORT 8080
ENV MODEL_PATH models/allocation_model.pkl

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies first (for faster caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Ensure the model file permissions are correct (if needed)
# Since you're using LFS, the file should be there at this point.

# Expose the application port
EXPOSE $PORT

# Command to run the application using Procfile command
# Note: The Procfile is still useful, but we can set the command directly here for full control.
CMD gunicorn --bind 0.0.0.0:$PORT api:app