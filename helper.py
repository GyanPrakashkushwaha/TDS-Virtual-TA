import time

class RateLimiter:
    def __init__(self, requests_per_minute: int = 300, requests_per_second: int = 100000):
        self.requests_per_minute = requests_per_minute
        self.requests_per_second = requests_per_second
        self.request_times = []
        self.last_request_time = 0
    
    def wait(self):
        curr_time = time.time()
        
        # per-second rate limiting
        time_since_last_request = curr_time - self.last_request_time
        if time_since_last_request < (1 / self.requests_per_second):
            time.sleep((1 / self.requests_per_second) - time_since_last_request)
        
        # per-minute rate limiting
        curr_time = time.time()
        self.request_times = [t for t in self.request_times if t > curr_time - 60]
        if len(self.request_times) >= self.requests_per_minute:
            sleep_time = 60 - (curr_time - self.request_times[0])
            if sleep_time > 0:
                time.sleep(sleep_time)
                curr_time = time.time()
                self.request_times = [t for t in self.request_times if t > curr_time - 60]
                
        self.request_times.append(curr_time)
        self.last_request_time = curr_time
