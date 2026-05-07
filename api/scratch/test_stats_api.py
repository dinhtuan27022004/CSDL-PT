import requests
import json

def test():
    base_url = "http://localhost:8000/api/data/stats"
    
    print("Fetching FULL stats...")
    full_res = requests.get(f"{base_url}?mode=full").json()
    print(f"Full Count: {full_res['count']}")
    print(f"Full Unique Images: {full_res['coverage']['unique_images']}")
    
    print("\nFetching DIVERSE stats...")
    diverse_res = requests.get(f"{base_url}?mode=diverse").json()
    print(f"Diverse Count: {diverse_res['count']}")
    print(f"Diverse Unique Images: {diverse_res['coverage']['unique_images']}")

if __name__ == "__main__":
    test()
