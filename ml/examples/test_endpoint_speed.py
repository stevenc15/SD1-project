import time
import requests

def measure_speed(endpoint, num_requests=1000):
    response_times = []

    for _ in range(num_requests):
        start_time = time.time()
        try:
            response = requests.get(endpoint)
            response.raise_for_status()  # rause an error for bad responses
            elapsed_time = time.time() - start_time
            response_times.append(elapsed_time)
        except requests.RequestException as e:
            print(f"Request failed: {e}")

    average_response_time = sum(response_times) / len(response_times)
    print(f"Average response time over {num_requests} requests: {average_response_time:.6f} seconds")

if __name__ == "__main__":
    endpoint = "https://example.com/api"  # Replace with your endpoint
    measure_speed(endpoint)
