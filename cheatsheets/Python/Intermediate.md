## 2. PYTHON INTERMEDIATE CHEATSHEET

### Object-Oriented Programming
```python
# Basic class
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def greet(self):
        return f"Hi, I'm {self.name}"
    
    def __str__(self):
        return f"Person(name='{self.name}', age={self.age})"

# Class with properties
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2

# Inheritance
class Student(Person):
    def __init__(self, name, age, student_id):
        super().__init__(name, age)
        self.student_id = student_id
    
    def study(self, subject):
        return f"{self.name} is studying {subject}"
    
    def greet(self):  # Method overriding
        return f"Hi, I'm {self.name}, student ID: {self.student_id}"

# Class methods and static methods
class MathUtils:
    pi = 3.14159
    
    @classmethod
    def circle_area(cls, radius):
        return cls.pi * radius ** 2
    
    @staticmethod
    def add(a, b):
        return a + b
```

### Advanced Data Structures
```python
# Collections module
from collections import defaultdict, Counter, deque, namedtuple

# defaultdict
dd = defaultdict(list)
dd['key1'].append('value1')  # No KeyError

# Counter
text = "hello world"
counter = Counter(text)
print(counter.most_common(3))  # Most frequent characters

# deque (double-ended queue)
dq = deque(['a', 'b', 'c'])
dq.appendleft('z')    # Add to left
dq.append('d')        # Add to right
left_item = dq.popleft()   # Remove from left

# namedtuple
Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)
print(p.x, p.y)
```

### Iterators and Generators
```python
# Custom iterator
class Countdown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        self.start -= 1
        return self.start + 1

# Generator function
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# Generator expressions
squares = (x**2 for x in range(10))
even_squares = (x**2 for x in range(10) if x % 2 == 0)

# Using generators
fib_gen = fibonacci(10)
for num in fib_gen:
    print(num)
```

### Decorators
```python
# Simple decorator
def timer(func):
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)
    return "Done"

# Decorator with parameters
def retry(max_attempts):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise e
                    print(f"Attempt {attempt + 1} failed: {e}")
        return wrapper
    return decorator

@retry(max_attempts=3)
def unreliable_function():
    import random
    if random.random() < 0.7:
        raise Exception("Random failure")
    return "Success"

# Class decorators
def add_method(cls):
    def new_method(self):
        return "Added method"
    cls.new_method = new_method
    return cls

@add_method
class MyClass:
    pass
```

### Context Managers
```python
# Custom context manager
class DatabaseConnection:
    def __enter__(self):
        print("Connecting to database")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing database connection")
        return False  # Don't suppress exceptions
    
    def query(self, sql):
        return f"Results for: {sql}"

# Using context manager
with DatabaseConnection() as db:
    result = db.query("SELECT * FROM users")

# Context manager with contextlib
from contextlib import contextmanager

@contextmanager
def temporary_value(obj, attr, value):
    old_value = getattr(obj, attr)
    setattr(obj, attr, value)
    try:
        yield obj
    finally:
        setattr(obj, attr, old_value)
```

### Regular Expressions
```python
import re

# Basic patterns
text = "Contact: john@example.com or call 123-456-7890"

# Find email
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
email = re.search(email_pattern, text)
if email:
    print(f"Email: {email.group()}")

# Find all phone numbers
phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
phones = re.findall(phone_pattern, text)

# Replace patterns
cleaned_text = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', text)

# Compile patterns for reuse
email_regex = re.compile(email_pattern)
matches = email_regex.findall(text)

# Groups and named groups
pattern = r'(?P<area>\d{3})-(?P<exchange>\d{3})-(?P<number>\d{4})'
match = re.search(pattern, text)
if match:
    print(match.groupdict())  # {'area': '123', 'exchange': '456', 'number': '7890'}
```

### File and Directory Operations
```python
import os
import shutil
from pathlib import Path

# os module
current_dir = os.getcwd()
os.makedirs('new_directory', exist_ok=True)
os.rename('old_name.txt', 'new_name.txt')

# List directory contents
for item in os.listdir('.'):
    if os.path.isfile(item):
        print(f"File: {item}")
    elif os.path.isdir(item):
        print(f"Directory: {item}")

# shutil for high-level operations
shutil.copy('source.txt', 'destination.txt')
shutil.move('file.txt', 'new_location/')

# pathlib (modern approach)
path = Path('documents/file.txt')
print(path.parent)      # documents
print(path.name)        # file.txt
print(path.suffix)      # .txt
print(path.exists())    # True/False

# Create directories and files
Path('new_dir').mkdir(parents=True, exist_ok=True)
Path('new_file.txt').touch()

# Glob patterns
for txt_file in Path('.').glob('*.txt'):
    print(txt_file)
```

### Working with JSON and CSV
```python
import json
import csv

# JSON operations
data = {'name': 'Alice', 'age': 25, 'city': 'New York'}

# Write JSON
with open('data.json', 'w') as f:
    json.dump(data, f, indent=2)

# Read JSON
with open('data.json', 'r') as f:
    loaded_data = json.load(f)

# JSON strings
json_string = json.dumps(data)
parsed_data = json.loads(json_string)

# CSV operations
data = [
    ['Name', 'Age', 'City'],
    ['Alice', 25, 'New York'],
    ['Bob', 30, 'Boston']
]

# Write CSV
with open('people.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(data)

# Read CSV
with open('people.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)

# CSV with dictionaries
with open('people.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"Name: {row['Name']}, Age: {row['Age']}")
```

### Threading and Multiprocessing
```python
import threading
import multiprocessing
import time

# Threading
def worker(name):
    for i in range(5):
        print(f"Worker {name}: {i}")
        time.sleep(1)

# Create and start threads
threads = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    threads.append(t)
    t.start()

# Wait for all threads to complete
for t in threads:
    t.join()

# Thread with shared data
import threading

counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

# Multiprocessing
def cpu_bound_task(n):
    total = 0
    for i in range(n):
        total += i * i
    return total

# Process pool
with multiprocessing.Pool() as pool:
    results = pool.map(cpu_bound_task, [100000, 200000, 300000])
    print(results)
```