import modal
from modal import Image

# Setup

app = modal.App("hello")
image = Image.debian_slim().pip_install("requests")

# Hello functions for different regions


@app.function(image=image)
def hello() -> str:
    """Default hello function - runs in default region"""
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]       
    return f"Hello from {city}, {region}, {country}!!"


@app.function(image=image, region="eu")
def hello_europe() -> str:
    """Hello function for Europe region"""
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]       
    return f"Hello from {city}, {region}, {country}!!"


@app.function(image=image, region="in")
def hello_india() -> str:
    """Hello function for India region"""
    import requests

    response = requests.get("https://ipinfo.io/json")
    data = response.json()
    city, region, country = data["city"], data["region"], data["country"]       
    return f"Hello from {city}, {region}, {country}!!"
