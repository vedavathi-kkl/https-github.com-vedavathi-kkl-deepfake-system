import requests

url = "https://thispersondoesnotexist.com/image"
r = requests.get(url)

print("Status:", r.status_code)