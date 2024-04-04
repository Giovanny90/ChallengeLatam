from challenge.api import app
import uvicorn
application = app

if __name__ == "__main__":
    uvicorn.run(application, host="0.0.0.0", port=8000)