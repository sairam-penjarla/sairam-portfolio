> Welcome to your Python programming journey! If you're an absolute beginner, don't worry – we’ll take it step by step. Python is one of the most beginner-friendly programming languages, and by the end of this guide, you'll be able to write your own Python scripts, solve problems, and create projects.
> 

# Table of Contents

# Why Python?

Python is great for beginners because:

1. It has simple and readable syntax.
2. It can be used for a variety of applications, from web development to data analysis, machine learning, and automation.
3. The Python community is vast and supportive, so help is never far away.

In this guide, we'll cover the basics of Python programming in a structured manner, with practice code examples, explanations, and the motivation for hands-on practice. We'll also encourage you to use Google Colab to experiment with the code yourself.

# Setup python:

We have provided step-by-step instructions to guide you and ensure a successful installation. Whether you are new to programming or have some experience, mastering how to install Python on Windows will enable you to utilize this potent language and uncover its full range of potential applications.

To download Python on your system, you can use the following steps

## **Step 1: Select Version to Install Python**

Navigate to the official Python page at https://www.python.org/downloads/ using an operating system. Look for a stable version of Python 3, ideally version 3.10.11, which is the version used in testing for this tutorial. Choose the appropriate link for your device from the available options: either "Windows installer (64-bit)" or "Windows installer (32-bit)", and proceed to download the executable file.

![](https://static.wixstatic.com/media/d4dfda_32fabed4f64c4e6694ec996453115872~mv2.png/v1/fill/w_1480,h_694,al_c,q_90,usm_0.66_1.00_0.01,enc_auto/d4dfda_32fabed4f64c4e6694ec996453115872~mv2.png)

## **Step 2: Downloading the Python Installer**

After downloading the installer, open the .exe file (e.g., python-3.10.11-amd64.exe) by double-clicking it to start the Python installer. During installation, ensure all users of the computer can access the Python launcher application by selecting the option to install the launcher for all users.

<aside>
💡

**Additionally, don’t forget to check the "Add python.exe to PATH" checkbox.**

</aside>

![](https://static.wixstatic.com/media/d4dfda_3c4b7c6526224e6c8f538ca35e6dcc02~mv2.png/v1/fill/w_1480,h_914,al_c,q_90,usm_0.66_1.00_0.01,enc_auto/d4dfda_3c4b7c6526224e6c8f538ca35e6dcc02~mv2.png)

**After Clicking the Install Now Button the setup will start installing Python on your Windows system.**

# **Module 3**

## 1. **Decorators**

Decorators are a powerful tool in Python that allows you to modify the behavior of a function or class. A decorator is essentially a function that takes another function as an argument and returns a new function that enhances or modifies the original function’s behavior.

### Defining a Simple Decorator

```python
def decorator_function(original_function):
    def wrapper_function():
        print("Wrapper executed before the function.")
        return original_function()
    return wrapper_function

def display():
    return "Display function executed!"

decorated_display = decorator_function(display)
print(decorated_display())  # Output: Wrapper executed before the function.
                           #         Display function executed!

```

### Using `@` Syntax for Decorators

The `@decorator_name` syntax is a more readable and concise way to apply decorators to functions.

```python
def decorator_function(original_function):
    def wrapper_function():
        print("Wrapper executed before the function.")
        return original_function()
    return wrapper_function

@decorator_function
def display():
    return "Display function executed!"

print(display())  # Output: Wrapper executed before the function.
                  #         Display function executed!

```

### Decorators with Arguments

Decorators can also accept arguments by using an additional nested function.

```python
def decorator_with_args(argument):
    def decorator_function(original_function):
        def wrapper_function(*args, **kwargs):
            print(f"Decorator argument: {argument}")
            return original_function(*args, **kwargs)
        return wrapper_function
    return decorator_function

@decorator_with_args("Test argument")
def display():
    return "Display function executed!"

print(display())  # Output: Decorator argument: Test argument
                  #         Display function executed!

```

---

## 2. **Generators**

Generators are a type of iterable, like lists, but instead of storing all values in memory, they generate values on the fly. You can create generators using functions with the `yield` keyword.

### Defining a Generator Function

```python
def count_up_to(limit):
    count = 1
    while count <= limit:
        yield count
        count += 1

counter = count_up_to(5)
for number in counter:
    print(number)

```

**Output**:

```
1
2
3
4
5

```

### The `yield` Keyword

The `yield` keyword returns a value from the generator function and suspends the function’s state. The next time the generator’s `__next__()` method is called, the function resumes from where it left off.

```python
def countdown(n):
    while n > 0:
        yield n
        n -= 1

counter = countdown(3)
print(next(counter))  # Output: 3
print(next(counter))  # Output: 2
print(next(counter))  # Output: 1

```

### Generator Expressions

You can also create generators using a shorthand syntax called a generator expression, similar to a list comprehension, but with parentheses instead of square brackets.

```python
gen = (x * x for x in range(1, 6))
for value in gen:
    print(value)

```

**Output**:

```
1
4
9
16
25

```

---

## 3. **File I/O (Input/Output)**

### Reading from a File

You can use the `open()` function to open a file and read its contents.

```python
# Open file in read mode
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

```

The `with` statement ensures that the file is properly closed after it is no longer needed, even if an error occurs.

### Writing to a File

You can write to a file using the `write()` or `writelines()` methods.

```python
# Open file in write mode (will overwrite if file exists)
with open("example.txt", "w") as file:
    file.write("Hello, this is a test file.")

# Open file in append mode (will add to the file)
with open("example.txt", "a") as file:
    file.write("\\nThis is an appended line.")

```

### Reading Line by Line

To read a file line by line, use the `readlines()` method or a loop.

```python
with open("example.txt", "r") as file:
    lines = file.readlines()
    for line in lines:
        print(line.strip())  # `strip()` removes newline characters

```

---

## 4. **Regular Expressions (Regex)**

Regular expressions are used for pattern matching and text manipulation. Python’s `re` module provides support for working with regular expressions.

### Searching for a Pattern

You can use the `re.search()` function to find a match for a pattern in a string.

```python
import re

pattern = r"hello"
text = "hello world"
match = re.search(pattern, text)

if match:
    print("Match found!")
else:
    print("No match found.")

```

### Matching Patterns

The `re.match()` function checks if the pattern matches from the beginning of the string.

```python
import re

pattern = r"hello"
text = "hello world"
match = re.match(pattern, text)

if match:
    print("Match found at the start!")
else:
    print("No match found at the start.")

```

### Substituting Text

You can use `re.sub()` to replace parts of the string that match the pattern.

```python
import re

text = "The sky is blue."
new_text = re.sub(r"blue", "green", text)
print(new_text)  # Output: The sky is green.

```

---

## 5. **Multithreading**

Multithreading is a way to run multiple tasks concurrently. Python provides the `threading` module to work with threads.

### Creating and Starting Threads

```python
import threading

def print_numbers():
    for i in range(1, 6):
        print(i)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()  # Wait for the thread to finish

```

### Using `ThreadPoolExecutor`

`ThreadPoolExecutor` is an easier way to manage a pool of threads for concurrent execution.

```python
from concurrent.futures import ThreadPoolExecutor

def print_numbers():
    for i in range(1, 6):
        print(i)

with ThreadPoolExecutor() as executor:
    executor.submit(print_numbers)

```

---

## 6. **Context Managers**

Context managers are used to manage resources, such as files, database connections, or network connections, ensuring that they are properly cleaned up when done.

### Using `with` Statement

The `with` statement simplifies resource management by automatically handling setup and teardown operations.

```python
with open("example.txt", "r") as file:
    content = file.read()
    print(content)

```

### Custom Context Managers

You can create custom context managers using the `contextlib` module or by defining a class with `__enter__()` and `__exit__()` methods.

```python
class MyContextManager:
    def __enter__(self):
        print("Entering the context")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting the context")

with MyContextManager():
    print("Inside the context")

```

**Output**:

```
Entering the context
Inside the context
Exiting the context

```

---

## 7. **Testing in Python**

### Using `unittest`

The `unittest` module provides a framework for writing and running tests.

```python
import unittest

def add(x, y):
    return x + y

class TestAddition(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)

if __name__ == "__main__":
    unittest.main()

```

### Using `pytest`

`pytest` is a popular testing framework in Python. It offers a simple syntax for writing test functions.

```python
# test_example.py
def add(x, y):
    return x + y

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0

# Run pytest in the terminal
# $ pytest test_example.py

```

---

## 8. **Concurrency: Asyncio**

Asyncio is a Python library used for writing concurrent code using asynchronous I/O operations. It is particularly useful for I/O-bound tasks like network requests, file I/O, or database queries.

### Basic Asyncio Example

```python
import asyncio

async def greet():
    print("Hello,")
    await asyncio.sleep(1)
    print("World!")

async def main():
    await greet()

asyncio.run(main())

```

In the above example, the `async` keyword defines an asynchronous function, and `await` is used to pause execution until the asynchronous task completes.

### Running Multiple Tasks Concurrently

```python
async def task1():
    await asyncio.sleep(1)
    print("Task 1 completed")

async def task2():
    await asyncio.sleep(2)
    print("Task 2 completed")

async def main():
    await asyncio.gather(task1(), task2())

asyncio.run(main())

```

---

## 9. **Memory Management and Garbage Collection**

Python uses automatic memory management and garbage collection to manage the allocation and deallocation of memory.

### Reference Counting

Python uses reference counting to keep track of how many references exist to an object. When an object’s reference count drops to zero, it is deallocated.

```python
import sys

x = [1, 2, 3]
print(sys.getrefcount(x))  # Output: Number of references to the object

```

### Garbage Collection

Python also has a garbage collector that detects and cleans up cyclic references, which cannot be handled by reference counting alone.

You can interact with the garbage collector using the `gc` module.

```python
import gc
gc.collect()  # Force garbage collection

```

---

## 10. **Metaclasses**

Metaclasses are classes of classes. They are used to control the creation and behavior of classes in Python.

### Defining a Simple Metaclass

```python
class MyMeta(type):
    def __new__(cls, name, bases, dct):
        dct['class_name'] = name
        return super().__new__(cls, name, bases, dct)

class MyClass(metaclass=MyMeta):
    pass

print(MyClass.class_name)  # Output: MyClass

```

In the above example, the metaclass `MyMeta` modifies the class creation by adding a `class_name` attribute to it.

### Metaclass for Singleton Pattern

```python
class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class SingletonClass(metaclass=SingletonMeta):
    pass

obj1 = SingletonClass()
obj2 = SingletonClass()
print(obj1 is obj2)  # Output: True

```

---

## 11. **Profiling and Optimization**

Profiling helps you understand the performance bottlenecks in your code, and optimization allows you to improve the performance.

### Using `cProfile` for Profiling

You can use the `cProfile` module to profile your Python programs.

```python
import cProfile

def slow_function():
    for _ in range(1000000):
        pass

cProfile.run('slow_function()')

```

This will output profiling data, showing the time spent in different parts of your code.

### Using `timeit` for Microbenchmarking

You can use the `timeit` module to measure the execution time of small code snippets.

```python
import timeit

print(timeit.timeit('x = [i for i in range(1000)]', number=1000))  # Time to run the code 1000 times

```

---

## 12. **Unit Testing**

Unit testing is a crucial part of ensuring your code works as expected. Python provides the `unittest` framework to write and run tests for your functions and classes.

### Writing Unit Tests

```python
import unittest

def add(x, y):
    return x + y

class TestMathOperations(unittest.TestCase):
    def test_add(self):
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(add(-1, 1), 0)
        self.assertEqual(add(0, 0), 0)

if __name__ == '__main__':
    unittest.main()

```

In the above code, the `TestMathOperations` class inherits from `unittest.TestCase`. The `test_add` method is a unit test that checks if the `add` function works correctly.

### Running Tests

You can run the tests from the command line using:

```bash
python -m unittest test_module.py

```

---

Python offers both threading and multiprocessing for concurrent execution of tasks.

### Threading

Threading is used to run tasks concurrently in the same process. It is suitable for I/O-bound tasks.

```python
import threading

def print_numbers():
    for i in range(5):
        print(i)

thread = threading.Thread(target=print_numbers)
thread.start()
thread.join()  # Wait for the thread to finish

```

## 13. Multiprocessing

Multiprocessing is used for CPU-bound tasks and allows running tasks in separate processes, utilizing multiple CPU cores.

```python
import multiprocessing

def square_number(n):
    return n * n

if __name__ == '__main__':
    with multiprocessing.Pool() as pool:
        results = pool.map(square_number, [1, 2, 3, 4, 5])
    print(results)  # Output: [1, 4, 9, 16, 25]

```

`multiprocessing.Pool` is used to parallelize tasks across multiple processes.

## 14. **Serialization with `pickle`**

Serialization allows you to save Python objects to files and load them back later. The `pickle` module provides a way to serialize and deserialize Python objects.

### Pickling and Unpickling Objects

```python
import pickle

# Pickling
data = {'name': 'Alice', 'age': 25}
with open('data.pickle', 'wb') as f:
    pickle.dump(data, f)

# Unpickling
with open('data.pickle', 'rb') as f:
    loaded_data = pickle.load(f)
print(loaded_data)  # Output: {'name': 'Alice', 'age': 25}

```

Pickle allows storing and retrieving complex objects like dictionaries, lists, and custom classes.

---

## 15. **Logging**

The `logging` module in Python is used for logging messages from your code, which can help with debugging and monitoring.

### Basic Logging

```python
import logging

logging.basicConfig(level=logging.DEBUG)
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")

```

You can configure the logging level to filter messages based on severity (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).

---

## 16. **Command-Line Arguments**

Python allows you to pass arguments to your script using the `sys.argv` list or the `argparse` module.

### Using `sys.argv`

```python
import sys

print(f"Script name: {sys.argv[0]}")
for i in range(1, len(sys.argv)):
    print(f"Argument {i}: {sys.argv[i]}")

```

Running the script with arguments:

```bash
python myscript.py arg1 arg2

```

### Using `argparse`

```python
import argparse

parser = argparse.ArgumentParser(description="My Script")
parser.add_argument('name', help='Your name')
parser.add_argument('age', type=int, help='Your age')
args = parser.parse_args()

print(f"Hello, {args.name}! You are {args.age} years old.")

```

You can pass arguments to the script like:

```bash
python myscript.py Alice 30

```

---

### **Conclusion: Practice Makes Perfect!**

Now that you’ve learned the basics of Python, the key to mastering programming is practice. Try solving problems on platforms like [**CodeChef**](https://www.codechef.com/practice-old) and learn about [**Data Structures and Algorithms**](https://www.geeksforgeeks.org/data-structures/) to improve your skills further.

Don't forget, the best way to learn is by doing. Use **Google Colab** to write, test, and run your Python code. You'll find that hands-on practice is the most important step in becoming proficient in Python.

Happy coding, and keep practicing!