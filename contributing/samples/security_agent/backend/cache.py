from cachetools import TTLCache
from functools import wraps

# In-memory cache with a TTL of 10 minutes
cache = TTLCache(maxsize=100, ttl=600)

def cached(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Create a cache key from the function name and arguments
        key = f"{func.__name__}:{args}:{kwargs}"
        
        # Check if the result is in the cache
        if key in cache:
            return cache[key]
        
        # If not, call the function and store the result in the cache
        result = func(*args, **kwargs)
        cache[key] = result
        return result
    return wrapper
