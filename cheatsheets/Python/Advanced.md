## 3. PYTHON ADVANCED CHEATSHEET

### Metaclasses
```python
# Simple metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Singleton(metaclass=SingletonMeta):
    def __init__(self):
        self.value = 0

# Metaclass with __new__
class AttributeValidatorMeta(type):
    def __new__(mcs, name, bases, attrs):
        # Add validation to all methods
        for attr_name, attr_value in list(attrs.items()):
            if callable(attr_value) and not attr_name.startswith('_'):
                attrs[attr_name] = mcs.validate_method(attr_value)
        return super().__new__(mcs, name, bases, attrs)
    
    @staticmethod
    def validate_method(method):
        def wrapper(*args, **kwargs):
            print(f"Calling {method.__name__}")
            return method(*args, **kwargs)
        return wrapper

class MyClass(metaclass=AttributeValidatorMeta):
    def my_method(self):
        return "Hello"
```

### Descriptors
```python
# Property descriptor
class ValidatedAttribute:
    def __init__(self, name, validator):
        self.name = name
        self.validator = validator
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj.__dict__.get(self.name)
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError(f"Invalid value for {self.name}")
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        del obj.__dict__[self.name]

# Usage
class Person:
    name = ValidatedAttribute('name', lambda x: isinstance(x, str) and len(x) > 0)
    age = ValidatedAttribute('age', lambda x: isinstance(x, int) and 0 <= x <= 150)
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Non-data descriptor
class CachedProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        value = self.func(obj)
        obj.__dict__[self.name] = value  # Cache the value
        return value

class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @CachedProperty
    def area(self):
        print("Calculating area...")  # This will only print once
        return 3.14159 * self.radius ** 2
```

### Advanced Function Features
```python
# Function annotations
def process_data(data: list[int], multiplier: float = 1.0) -> list[float]:
    """Process a list of integers with a multiplier."""
    return [x * multiplier for x in data]

# Introspection
import inspect

def analyze_function(func):
    sig = inspect.signature(func)
    print(f"Function: {func.__name__}")
    print(f"Parameters: {list(sig.parameters.keys())}")
    print(f"Annotations: {func.__annotations__}")
    print(f"Docstring: {func.__doc__}")

# Partial functions
from functools import partial

def multiply(x, y, z):
    return x * y * z

double = partial(multiply, 2)  # Fix first argument
result = double(3, 4)  # Same as multiply(2, 3, 4)

# Function caching
from functools import lru_cache, cache

@lru_cache(maxsize=128)
def expensive_function(n):
    # Simulate expensive computation
    import time
    time.sleep(1)
    return n ** 2

@cache  # Unlimited cache (Python 3.9+)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Function dispatch
from functools import singledispatch

@singledispatch
def process(arg):
    print(f"Processing {type(arg)}: {arg}")

@process.register
def _(arg: int):
    print(f"Processing integer: {arg * 2}")

@process.register
def _(arg: str):
    print(f"Processing string: {arg.upper()}")
```

### Advanced Async Programming
```python
import asyncio
import aiohttp
import aiofiles

# Basic async function
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Async file operations
async def read_file_async(filename):
    async with aiofiles.open(filename, 'r') as f:
        content = await f.read()
        return content

# Async generators
async def async_fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        await asyncio.sleep(0.1)  # Simulate async work
        a, b = b, a + b

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        await asyncio.sleep(0.1)

# Running async code
async def main():
    # Gather multiple async operations
    urls = ['http://example.com', 'http://google.com']
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    
    # Async iteration
    async for fib_num in async_fibonacci(5):
        print(fib_num)
    
    # Async context manager
    async with AsyncResource() as resource:
        print("Using resource")

# Run the async program
# asyncio.run(main())
```

### Memory Management and Optimization
```python
# Weak references
import weakref

class Parent:
    def __init__(self, name):
        self.name = name
        self.children = []

class Child:
    def __init__(self, name, parent):
        self.name = name
        self.parent = weakref.ref(parent)  # Weak reference to avoid circular ref
        parent.children.append(self)

# Memory profiling
import sys
import gc
from memory_profiler import profile

@profile
def memory_intensive_function():
    # Create large list
    data = [i for i in range(1000000)]
    return sum(data)

# Object size
def get_object_size(obj):
    return sys.getsizeof(obj)

# Garbage collection
def cleanup_memory():
    collected = gc.collect()
    print(f"Collected {collected} objects")

# Slots for memory efficiency
class OptimizedClass:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
```

### Advanced Data Manipulation
```python
# Custom data structures
class Trie:
    def __init__(self):
        self.children = {}
        self.is_end = False
    
    def insert(self, word):
        node = self
        for char in word:
            if char not in node.children:
                node.children[char] = Trie()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

# Custom operators
class Vector:
    def __init__(self, *components):
        self.components = list(components)
    
    def __add__(self, other):
        return Vector(*[a + b for a, b in zip(self.components, other.components)])
    
    def __mul__(self, scalar):
        return Vector(*[scalar * component for component in self.components])
    
    def __matmul__(self, other):  # @ operator
        return sum(a * b for a, b in zip(self.components, other.components))
    
    def __repr__(self):
        return f"Vector{tuple(self.components)}"

# Usage
v1 = Vector(1, 2, 3)
v2 = Vector(4, 5, 6)
print(v1 + v2)      # Vector(5, 7, 9)
print(v1 * 2)       # Vector(2, 4, 6)
print(v1 @ v2)      # 32 (dot product)
```

### Testing and Debugging
```python
import unittest
import pytest
import logging
from unittest.mock import Mock, patch

# Unit testing with unittest
class TestMathOperations(unittest.TestCase):
    def setUp(self):
        self.calculator = Calculator()
    
    def test_addition(self):
        result = self.calculator.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_division_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calculator.divide(5, 0)

# Pytest style
def test_string_operations():
    text = "hello world"
    assert text.upper() == "HELLO WORLD"
    assert len(text.split()) == 2

# Mocking
class APIClient:
    def get_user(self, user_id):
        # Simulate API call
        pass

def test_api_client():
    client = APIClient()
    
    # Mock the API call
    with patch.object(client, 'get_user') as mock_get_user:
        mock_get_user.return_value = {'id': 1, 'name': 'Alice'}
        
        result = client.get_user(1)
        assert result['name'] == 'Alice'
        mock_get_user.assert_called_once_with(1)

# Logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def debug_function():
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")

# Custom exception with traceback
import traceback

def handle_error():
    try:
        raise ValueError("Something went wrong")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        logger.error(traceback.format_exc())
```

### Performance and Profiling
```python
import timeit
import cProfile
import pstats
from functools import wraps

# Timing code
def time_function():
    # Time a simple operation
    time_taken = timeit.timeit('sum([1, 2, 3, 4, 5])', number=100000)
    print(f"Time taken: {time_taken}")

# Profiling decorator
def profile_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = func(*args, **kwargs)
        pr.disable()
        
        stats = pstats.Stats(pr)
        stats.sort_stats('cumulative')
        stats.print_stats(10)  # Top 10 functions
        
        return result
    return wrapper

@profile_function
def slow_function():
    total = 0
    for i in range(100000):
        total += i ** 2
    return total

# Line profiler (requires line_profiler package)
@profile  # This decorator is provided by kernprof
def line_by_line_profile():
    data = []
    for i in range(1000):
        data.append(i ** 2)
    
    result = sum(data)
    return result

# Memory optimization techniques
def generator_vs_list():
    # Memory efficient generator
    def fibonacci_gen(n):
        a, b = 0, 1
        for _ in range(n):
            yield a
            a, b = b, a + b
    
    # Memory intensive list
    def fibonacci_list(n):
        fib_list = []
        a, b = 0, 1
        for _ in range(n):
            fib_list.append(a)
            a, b = b, a + b
        return fib_list
```

### Type Hints and Static Analysis
```python
from typing import (
    List, Dict, Tuple, Optional, Union, Callable, 
    TypeVar, Generic, Protocol, Literal, Final
)

# Basic type hints
def process_items(items: List[str]) -> Dict[str, int]:
    return {item: len(item) for item in items}

# Generic types
T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Protocol for structural subtyping
class Drawable(Protocol):
    def draw(self) -> None: ...

def render_shape(shape: Drawable) -> None:
    shape.draw()  # Any object with draw() method works

# Advanced type hints
def complex_function(
    data: Dict[str, List[Tuple[int, str]]],
    callback: Callable[[str], bool],
    default: Optional[str] = None
) -> Union[str, None]:
    # Function implementation
    pass

# Literal types
def set_color(color: Literal['red', 'green', 'blue']) -> None:
    print(f"Setting color to {color}")

# Final variables
API_VERSION: Final = "1.0"

# Type aliases
UserId = int
UserData = Dict[str, Union[str, int]]
UserDatabase = Dict[UserId, UserData]

# Overloads
from typing import overload

@overload
def process(x: int) -> int: ...

@overload
def process(x: str) -> str: ...

def process(x):
    return x
```

### Concurrency Patterns
```python
import asyncio
import concurrent.futures
import threading
import queue
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

# Producer-Consumer pattern
class ProducerConsumer:
    def __init__(self, max_queue_size: int = 10):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
    
    def producer(self, items):
        for item in items:
            if self.stop_event.is_set():
                break
            self.queue.put(item)
            print(f"Produced: {item}")
        self.queue.put(None)  # Sentinel value
    
    def consumer(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                if item is None:  # Sentinel value
                    break
                print(f"Consumed: {item}")
                self.queue.task_done()
            except queue.Empty:
                continue

# Async semaphore for rate limiting
async def rate_limited_requests():
    semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
    
    async def make_request(url: str) -> str:
        async with semaphore:
            # Simulate HTTP request
            await asyncio.sleep(1)
            return f"Response from {url}"
    
    urls = [f"https://api.example.com/endpoint/{i}" for i in range(20)]
    tasks = [make_request(url) for url in urls]
    results = await asyncio.gather(*tasks)
    return results

# Thread pool with futures
def cpu_intensive_task(n: int) -> int:
    return sum(i * i for i in range(n))

def parallel_processing():
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        # Submit multiple tasks
        futures = [executor.submit(cpu_intensive_task, i * 10000) 
                  for i in range(1, 6)]
        
        # Get results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            print(f"Result: {result}")

# Async context manager for resource pooling
class AsyncResourcePool:
    def __init__(self, max_size: int = 10):
        self._pool = asyncio.Queue(maxsize=max_size)
        self._created = 0
        self._max_size = max_size
    
    async def acquire(self):
        try:
            return self._pool.get_nowait()
        except asyncio.QueueEmpty:
            if self._created < self._max_size:
                self._created += 1
                return self._create_resource()
            return await self._pool.get()
    
    async def release(self, resource):
        await self._pool.put(resource)
    
    def _create_resource(self):
        return f"Resource-{self._created}"

# Publish-Subscribe pattern
class EventBus:
    def __init__(self):
        self._subscribers = {}
    
    def subscribe(self, event_type: str, callback: Callable):
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(callback)
    
    def publish(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            for callback in self._subscribers[event_type]:
                callback(data)
    
    async def publish_async(self, event_type: str, data: Any):
        if event_type in self._subscribers:
            tasks = []
            for callback in self._subscribers[event_type]:
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(callback(data))
                else:
                    callback(data)
            if tasks:
                await asyncio.gather(*tasks)
```

### Advanced Algorithms and Data Structures
```python
# Decorator for memoization with TTL
import time
from functools import wraps

def memoize_with_ttl(ttl_seconds: int):
    def decorator(func):
        cache = {}
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            current_time = time.time()
            
            if key in cache:
                value, timestamp = cache[key]
                if current_time - timestamp < ttl_seconds:
                    return value
            
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        
        return wrapper
    return decorator

# Bloom filter implementation
import hashlib
from typing import Set

class BloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [False] * size
    
    def _hash(self, item: str, seed: int) -> int:
        hash_obj = hashlib.md5((item + str(seed)).encode())
        return int(hash_obj.hexdigest(), 16) % self.size
    
    def add(self, item: str):
        for i in range(self.hash_count):
            index = self._hash(item, i)
            self.bit_array[index] = True
    
    def might_contain(self, item: str) -> bool:
        for i in range(self.hash_count):
            index = self._hash(item, i)
            if not self.bit_array[index]:
                return False
        return True

# LRU Cache implementation
class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.usage_order = []
    
    def get(self, key):
        if key in self.cache:
            self.usage_order.remove(key)
            self.usage_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if key in self.cache:
            self.usage_order.remove(key)
        elif len(self.cache) >= self.capacity:
            oldest = self.usage_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = value
        self.usage_order.append(key)

# Binary search tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BST:
    def __init__(self):
        self.root = None
    
    def insert(self, val):
        self.root = self._insert_recursive(self.root, val)
    
    def _insert_recursive(self, node, val):
        if not node:
            return TreeNode(val)
        
        if val < node.val:
            node.left = self._insert_recursive(node.left, val)
        else:
            node.right = self._insert_recursive(node.right, val)
        
        return node
    
    def search(self, val):
        return self._search_recursive(self.root, val)
    
    def _search_recursive(self, node, val):
        if not node or node.val == val:
            return node
        
        if val < node.val:
            return self._search_recursive(node.left, val)
        return self._search_recursive(node.right, val)
    
    def inorder_traversal(self):
        result = []
        self._inorder_recursive(self.root, result)
        return result
    
    def _inorder_recursive(self, node, result):
        if node:
            self._inorder_recursive(node.left, result)
            result.append(node.val)
            self._inorder_recursive(node.right, result)

# Graph algorithms
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def add_edge(self, u, v):
        self.graph[u].append(v)
        self.graph[v].append(u)  # For undirected graph
    
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                queue.extend(self.graph[vertex])
        
        return result
    
    def dfs(self, start, visited=None):
        if visited is None:
            visited = set()
        
        visited.add(start)
        result = [start]
        
        for neighbor in self.graph[start]:
            if neighbor not in visited:
                result.extend(self.dfs(neighbor, visited))
        
        return result
    
    def dijkstra(self, start):
        import heapq
        
        distances = {node: float('inf') for node in self.graph}
        distances[start] = 0
        pq = [(0, start)]
        
        while pq:
            current_dist, current_node = heapq.heappop(pq)
            
            if current_dist > distances[current_node]:
                continue
            
            for neighbor in self.graph[current_node]:
                distance = current_dist + 1  # Assuming weight 1
                
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        
        return distances
```

### Database Integration
```python
import sqlite3
from contextlib import contextmanager
from typing import List, Dict, Any, Optional
import json

# Database context manager
@contextmanager
def get_db_connection(db_path: str):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable dict-like access
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# Database abstraction layer
class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        with get_db_connection(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    data JSON
                )
            ''')
    
    def create_user(self, name: str, email: str, data: Dict = None) -> int:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO users (name, email, data) VALUES (?, ?, ?)',
                (name, email, json.dumps(data) if data else None)
            )
            return cursor.lastrowid
    
    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'id': row['id'],
                    'name': row['name'],
                    'email': row['email'],
                    'data': json.loads(row['data']) if row['data'] else None
                }
        return None
    
    def update_user(self, user_id: int, **kwargs) -> bool:
        if not kwargs:
            return False
        
        set_clause = ', '.join(f'{key} = ?' for key in kwargs.keys())
        values = list(kwargs.values()) + [user_id]
        
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                f'UPDATE users SET {set_clause} WHERE id = ?',
                values
            )
            return cursor.rowcount > 0
    
    def delete_user(self, user_id: int) -> bool:
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute('DELETE FROM users WHERE id = ?', (user_id,))
            return cursor.rowcount > 0

# ORM-like pattern
class Model:
    table_name = None
    fields = []
    
    def __init__(self, **kwargs):
        for field in self.fields:
            setattr(self, field, kwargs.get(field))
    
    @classmethod
    def create_table(cls, db_path: str):
        # Simplified table creation
        pass
    
    def save(self, db_path: str):
        # Save instance to database
        pass
    
    @classmethod
    def find_by_id(cls, db_path: str, id: int):
        # Find record by ID
        pass

class User(Model):
    table_name = 'users'
    fields = ['id', 'name', 'email', 'created_at']
```

### Web Development Utilities
```python
import urllib.parse
import urllib.request
import json
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
import ssl

# HTTP client utilities
class HTTPClient:
    def __init__(self, base_url: str, headers: Dict[str, str] = None):
        self.base_url = base_url.rstrip('/')
        self.headers = headers or {}
    
    def get(self, endpoint: str, params: Dict = None) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if params:
            url += '?' + urllib.parse.urlencode(params)
        
        req = urllib.request.Request(url, headers=self.headers)
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    
    def post(self, endpoint: str, data: Dict) -> Dict:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        json_data = json.dumps(data).encode()
        
        headers = self.headers.copy()
        headers['Content-Type'] = 'application/json'
        
        req = urllib.request.Request(url, data=json_data, headers=headers)
        
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())

# Simple HTTP server
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True

class APIHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'status': 'OK', 'timestamp': time.time()}
            self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404)
    
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            data = json.loads(post_data.decode())
            # Process data here
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            response = {'received': data, 'status': 'processed'}
            self.wfile.write(json.dumps(response).encode())
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")

# URL routing system
class Router:
    def __init__(self):
        self.routes = {}
    
    def route(self, path: str, methods: List[str] = None):
        if methods is None:
            methods = ['GET']
        
        def decorator(func):
            for method in methods:
                key = f"{method}:{path}"
                self.routes[key] = func
            return func
        return decorator
    
    def dispatch(self, method: str, path: str, *args, **kwargs):
        key = f"{method}:{path}"
        if key in self.routes:
            return self.routes[key](*args, **kwargs)
        raise ValueError(f"No route found for {method} {path}")

# Usage example
router = Router()

@router.route('/users', ['GET', 'POST'])
def handle_users(request):
    if request.method == 'GET':
        return {'users': []}
    elif request.method == 'POST':
        return {'created': True}
```

### Configuration and Environment Management
```python
import os
import configparser
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import yaml

# Environment configuration
class Config:
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file
        self._config = {}
        self.load_config()
    
    def load_config(self):
        # Load from environment variables
        self._config.update(os.environ)
        
        # Load from config file if provided
        if self.config_file and Path(self.config_file).exists():
            if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                self.load_yaml_config()
            else:
                self.load_ini_config()
    
    def load_yaml_config(self):
        with open(self.config_file, 'r') as f:
            yaml_config = yaml.safe_load(f)
            self._flatten_dict(yaml_config)
    
    def load_ini_config(self):
        parser = configparser.ConfigParser()
        parser.read(self.config_file)
        
        for section in parser.sections():
            for key, value in parser.items(section):
                self._config[f"{section.upper()}_{key.upper()}"] = value
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}".upper() if parent_key else k.upper()
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        
        for key, value in items:
            self._config[key] = value
    
    def get(self, key: str, default: Any = None, cast_type: type = str):
        value = self._config.get(key.upper(), default)
        if value is not None and cast_type != str:
            try:
                if cast_type == bool:
                    return str(value).lower() in ('true', '1', 'yes', 'on')
                return cast_type(value)
            except (ValueError, TypeError):
                return default
        return value
    
    def get_int(self, key: str, default: int = 0) -> int:
        return self.get(key, default, int)
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        return self.get(key, default, bool)
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        return self.get(key, default, float)

# Settings dataclass
@dataclass
class AppSettings:
    debug: bool = False
    database_url: str = "sqlite:///app.db"
    api_key: str = ""
    max_workers: int = 4
    timeout: float = 30.0
    
    @classmethod
    def from_config(cls, config: Config):
        return cls(
            debug=config.get_bool('DEBUG'),
            database_url=config.get('DATABASE_URL', cls.database_url),
            api_key=config.get('API_KEY', ''),
            max_workers=config.get_int('MAX_WORKERS', cls.max_workers),
            timeout=config.get_float('TIMEOUT', cls.timeout)
        )

# Usage example
config = Config('config.yaml')
settings = AppSettings.from_config(config)
```

### Security and Cryptography
```python
import hashlib
import secrets
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

# Password hashing and verification
class PasswordManager:
    @staticmethod
    def hash_password(password: str, salt: bytes = None) -> tuple[str, str]:
        if salt is None:
            salt = secrets.token_bytes(32)
        
        # Use PBKDF2 with SHA256
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key.decode(), base64.urlsafe_b64encode(salt).decode()
    
    @staticmethod
    def verify_password(password: str, hashed_password: str, salt: str) -> bool:
        salt_bytes = base64.urlsafe_b64decode(salt.encode())
        key, _ = PasswordManager.hash_password(password, salt_bytes)
        return hmac.compare_digest(key, hashed_password)

# Encryption utilities
class EncryptionManager:
    def __init__(self, key: bytes = None):
        if key is None:
            key = Fernet.generate_key()
        self.cipher_suite = Fernet(key)
        self.key = key
    
    def encrypt(self, data: str) -> bytes:
        return self.cipher_suite.encrypt(data.encode())
    
    def decrypt(self, encrypted_data: bytes) -> str:
        return self.cipher_suite.decrypt(encrypted_data).decode()
    
    @classmethod
    def from_password(cls, password: str, salt: bytes = None):
        if salt is None:
            salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return cls(key)

# Secure token generation
class TokenGenerator:
    @staticmethod
    def generate_secure_token(length: int = 32) -> str:
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_hex_token(length: int = 32) -> str:
        return secrets.token_hex(length)
    
    @staticmethod
    def generate_numeric_token(length: int = 6) -> str:
        return ''.join(secrets.choice('0123456789') for _ in range(length))

# Rate limiting
class RateLimiter:
    def __init__(self, max_requests: int, time_window: int):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = {}
    
    def is_allowed(self, identifier: str) -> bool:
        import time
        current_time = time.time()
        
        if identifier not in self.requests:
            self.requests[identifier] = []
        
        # Remove old requests
        self.requests[identifier] = [
            req_time for req_time in self.requests[identifier]
            if current_time - req_time < self.time_window
        ]
        
        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            return False
        
        # Add current request
        self.requests[identifier].append(current_time)
        return True

# Input validation and sanitization
import re
from html import escape

class InputValidator:
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})
    PHONE_PATTERN = re.compile(r'^\+?1?[0-9]{10,15})
    
    @staticmethod
    def validate_email(email: str) -> bool:
        return bool(InputValidator.EMAIL_PATTERN.match(email))
    
    @staticmethod
    def validate_phone(phone: str) -> bool:
        return bool(InputValidator.PHONE_PATTERN.match(phone.replace('-', '').replace(' ', '')))
    
    @staticmethod
    def sanitize_html(text: str) -> str:
        return escape(text)
    
    @staticmethod
    def validate_length(text: str, min_length: int = 0, max_length: int = None) -> bool:
        if len(text) < min_length:
            return False
        if max_length and len(text) > max_length:
            return False
        return True
    
    @staticmethod
    def contains_sql_injection(text: str) -> bool:
        sql_patterns = [
            r'\bunion\b.*\bselect\b',
            r'\bor\b.*=.*=',
            r'\bdrop\b.*\btable\b',
            r'\bdelete\b.*\bfrom\b'
        ]
        text_lower = text.lower()
        return any(re.search(pattern, text_lower) for pattern in sql_patterns)
```