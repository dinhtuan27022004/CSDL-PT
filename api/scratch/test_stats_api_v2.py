import requests

def test():
    base_url = "http://localhost:8000/api/data/stats"
    
    for mode in ["full", "diverse"]:
        print(f"--- Testing {mode.upper()} ---")
        res = requests.get(f"{base_url}?mode={mode}").json()
        print(f"Count: {res['count']}")
        print(f"Unique Images: {res['coverage']['unique_images']}")
        print(f"Percentage: {res['coverage']['percentage']:.2f}%")
        print(f"Overlap Stats: {res['coverage']['overlap']}")
        print(f"Top Hub Count: {res['hub_images'][0]['count'] if res['hub_images'] else 0}")
        print("-" * 20)

if __name__ == "__main__":
    test()
