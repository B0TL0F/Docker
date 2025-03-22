import logging
import os

# Define log file name
LOG_FILE = os.path.join("logs","app.log")

# Ensure log file exists
if not os.path.exists(LOG_FILE):
    open(LOG_FILE, "w").close()

# Configure logging
logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s",  
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log to file
        logging.StreamHandler()         # Log to console
    ],
)

# Create a logger instance
logger = logging.getLogger("fastapi-app")
