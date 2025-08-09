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

# **Module 2:**

Now that you understand the basics, let’s dive into loops and functions, which are essential for writing efficient Python code.

## **1. For Loops**

A `for` loop is used to iterate over a sequence (like a list or a string).

```python
for x in range(6):
    if x == 3:
        break
    print(x)
else:
    print("Finally finished!")

```

**Output:**

```
0
1
2

```

The `else` block will NOT be executed if the loop is stopped by a `break` statement.

### **Nested Loops**

You can use loops within loops.

```python
adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for x in adj:
    for y in fruits:
        print(x, y)

```

**Output:**

```
red apple
red banana
red cherry
big apple
big banana
big cherry
tasty apple
tasty banana
tasty cherry

```

### **One-Line For Loop**

```python
[x for x in range(5)]

```

---

## **2. Functions**

Functions allow you to organize your code into reusable blocks. Here's a basic function:

```python
def greet(name):
    print("Hello, " + name)

greet("Alice")  # Output: Hello, Alice

```

---

## 3. Functions in Python

### Defining Functions

A function in Python is defined using the `def` keyword. You can define a function to perform specific tasks and optionally return values.

Example:

```python
def greet(name):
    return "Hello, " + name

result = greet("Alice")
print(result)  # Output: Hello, Alice

```

### Function Parameters and Arguments

Functions can take parameters, which are values passed into the function when it is called. You can also specify default values for parameters.

Example:

```python
def greet(name, age=25):
    return f"Hello, {name}! You are {age} years old."

print(greet("Alice"))  # Output: Hello, Alice! You are 25 years old.
print(greet("Bob", 30))  # Output: Hello, Bob! You are 30 years old.

```

### Return Statement

A function can return values to be used elsewhere in the program. If there is no `return` statement, the function returns `None` by default.

Example:

```python
def add(x, y):
    return x + y

result = add(5, 3)
print(result)  # Output: 8

```

### Function Scope

The scope of a variable refers to where it is accessible within the program. Variables defined inside a function are local to that function, and variables defined outside are global.

Example:

```python
x = "global"

def test_scope():
    x = "local"
    print("Inside function:", x)

test_scope()
print("Outside function:", x)

```

**Output**:

```
Inside function: local
Outside function: global

```

### **Built-in Functions**

- **`abs()`**: Returns the absolute value of a number.
- **`all()`**: Returns True if all elements in an iterable are true.
- **`any()`**: Returns True if any element in an iterable is true.
- **`bin()`**: Converts an integer to binary.

### Lambda Functions

Lambda functions are small anonymous functions that are defined using the `lambda` keyword. They are often used for simple operations that are used once or twice.

Example:

```python
multiply = lambda x, y: x * y
result = multiply(4, 5)
print(result)  # Output: 20

```

## 4. Modules and Packages

### Importing Modules

You can import external Python files (modules) to reuse their code in your program.

Example:

```python
import math
print(math.sqrt(16))  # Output: 4.0

```

### Aliasing Modules

You can use the `as` keyword to assign an alias to an imported module, which can make the code shorter and more readable.

Example:

```python
import math as m
print(m.pi)  # Output: 3.141592653589793

```

### From...Import Statement

You can import specific functions or variables from a module instead of importing the whole module.

Example:

```python
from math import sqrt
print(sqrt(25))  # Output: 5.0

```

### Creating Your Own Module

You can create your own module by saving your Python code in a `.py` file and importing it into other programs.

For example, create a file `mymodule.py` with the following content:

```python
def greet(name):
    return f"Hello, {name}!"

```

Then, in another Python file, you can import and use the function:

```python
import mymodule
print(mymodule.greet("Alice"))  # Output: Hello, Alice!

```

## 5. Exception Handling

### Try, Except Block

Exception handling allows you to catch errors and handle them gracefully instead of allowing the program to crash.

Example:

```python
try:
    x = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

```

**Output**:

```
Cannot divide by zero!

```

### Multiple Except Clauses

You can have multiple `except` clauses to handle different types of exceptions.

Example:

```python
try:
    x = int("hello")
except ValueError:
    print("Invalid value!")
except ZeroDivisionError:
    print("Cannot divide by zero!")

```

**Output**:

```
Invalid value!

```

### Finally Clause

The `finally` block will execute whether an exception occurs or not, making it useful for cleanup actions.

Example:

```python
try:
    file = open("file.txt", "r")
    # Read file content
except FileNotFoundError:
    print("File not found!")
finally:
    print("Execution finished.")

```

**Output**:

```
File not found!
Execution finished.

```

### Raising Exceptions

You can raise exceptions using the `raise` keyword, either for built-in exceptions or custom exceptions.

Example:

```python
def check_age(age):
    if age < 18:
        raise ValueError("Age must be 18 or older")
    else:
        print("Age is valid!")

try:
    check_age(16)
except ValueError as e:
    print(e)  # Output: Age must be 18 or older

```

## 6. Classes and Objects

### Defining a Class

A class is a blueprint for creating objects (instances). You define a class using the `class` keyword.

Example:

```python
class Dog:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def bark(self):
        return f"{self.name} says woof!"

# Create an object (instance) of the class
dog = Dog("Buddy", 3)
print(dog.bark())  # Output: Buddy says woof!

```

### The `__init__` Method

The `__init__` method is the constructor used for initializing new objects. It is called when a new instance of the class is created.

Example:

```python
class Car:
    def __init__(self, make, model, year):
        self.make = make
        self.model = model
        self.year = year

    def display_info(self):
        return f"{self.year} {self.make} {self.model}"

car = Car("Toyota", "Corolla", 2022)
print(car.display_info())  # Output: 2022 Toyota Corolla

```

### Inheritance

Inheritance allows a class to inherit attributes and methods from another class.

Example:

```python
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        return "Animal sound"

class Dog(Animal):
    def speak(self):
        return f"{self.name} says woof!"

dog = Dog("Buddy")
print(dog.speak())  # Output: Buddy says woof!

```

### Method Overriding

You can override methods in a subclass to provide a different implementation.

Example:

```python
class Animal:
    def speak(self):
        return "Animal sound"

class Dog(Animal):
    def speak(self):
        return "Woof!"

dog = Dog()
print(dog.speak())  # Output: Woof!

```

### Class and Instance Variables

- Class variables are shared by all instances of a class.
- Instance variables are unique to each instance of the class.

Example:

```python
class Cat:
    species = "Feline"  # Class variable

    def __init__(self, name):
        self.name = name  # Instance variable

cat1 = Cat("Whiskers")
cat2 = Cat("Tom")

print(cat1.name)     # Output: Whiskers
print(cat1.species)  # Output: Feline
print(cat2.name)     # Output: Tom
print(cat2.species)  # Output: Feline

```

### **Conclusion: Practice Makes Perfect!**

Now that you’ve learned the basics of Python, the key to mastering programming is practice. Try solving problems on platforms like [**CodeChef**](https://www.codechef.com/practice-old) and learn about [**Data Structures and Algorithms**](https://www.geeksforgeeks.org/data-structures/) to improve your skills further.

Don't forget, the best way to learn is by doing. Use **Google Colab** to write, test, and run your Python code. You'll find that hands-on practice is the most important step in becoming proficient in Python.

Happy coding, and keep practicing!