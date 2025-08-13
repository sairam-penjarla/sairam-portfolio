## 1. PYTHON BASICS CHEATSHEET

### Variables and Data Types
```python
# Basic data types
name = "Alice"          # String
age = 25               # Integer
height = 5.6           # Float
is_student = True      # Boolean
items = None           # NoneType

# Type checking and conversion
print(type(age))       # <class 'int'>
str_age = str(age)     # Convert to string
int_height = int(height)  # Convert to int
```

### Strings
```python
# String operations
text = "Hello World"
print(text.lower())         # hello world
print(text.upper())         # HELLO WORLD
print(text.strip())         # Remove whitespace
print(text.replace("o", "0"))  # Hell0 W0rld
print(text.split())         # ['Hello', 'World']

# String formatting
name = "Alice"
age = 25
print(f"My name is {name} and I'm {age}")  # f-strings
print("My name is {} and I'm {}".format(name, age))  # .format()
print("My name is %s and I'm %d" % (name, age))  # % formatting

# String methods
text = "python programming"
print(text.capitalize())    # Python programming
print(text.title())        # Python Programming
print(text.count('p'))     # 2
print(text.find('gram'))   # 10
print(text.startswith('py'))  # True
```

### Lists
```python
# List creation and operations
fruits = ['apple', 'banana', 'orange']
numbers = [1, 2, 3, 4, 5]
mixed = ['hello', 42, True, 3.14]

# List methods
fruits.append('grape')      # Add to end
fruits.insert(0, 'mango')  # Insert at index
fruits.remove('banana')    # Remove first occurrence
popped = fruits.pop()      # Remove and return last
fruits.extend(['kiwi', 'berry'])  # Add multiple items

# List operations
print(len(fruits))         # Length
print(fruits[0])          # First element
print(fruits[-1])         # Last element
print(fruits[1:3])        # Slice [index1:index2]
print('apple' in fruits)   # Check membership

# List comprehensions
squares = [x**2 for x in range(10)]
evens = [x for x in range(20) if x % 2 == 0]
```

### Tuples
```python
# Tuple creation (immutable)
coordinates = (3, 4)
colors = ('red', 'green', 'blue')
single_item = (42,)  # Note the comma

# Tuple operations
print(coordinates[0])      # Access by index
x, y = coordinates        # Tuple unpacking
print(len(colors))        # Length
print('red' in colors)    # Membership check
```

### Dictionaries
```python
# Dictionary creation and operations
person = {'name': 'Alice', 'age': 25, 'city': 'New York'}
person = dict(name='Alice', age=25, city='New York')  # Alternative

# Dictionary methods
print(person['name'])           # Access value
print(person.get('height', 0))  # Get with default
person['height'] = 5.6         # Add/update key
del person['city']             # Delete key

# Dictionary operations
print(person.keys())           # Get all keys
print(person.values())         # Get all values
print(person.items())          # Get key-value pairs
person.update({'age': 26, 'job': 'Engineer'})  # Update multiple

# Dictionary comprehension
squares = {x: x**2 for x in range(5)}
```

### Sets
```python
# Set creation and operations
fruits = {'apple', 'banana', 'orange'}
fruits = set(['apple', 'banana', 'orange'])  # From list

# Set methods
fruits.add('grape')           # Add element
fruits.remove('banana')       # Remove (raises error if not found)
fruits.discard('kiwi')       # Remove (no error if not found)
fruits.update(['mango', 'berry'])  # Add multiple

# Set operations
set1 = {1, 2, 3, 4}
set2 = {3, 4, 5, 6}
print(set1 | set2)           # Union
print(set1 & set2)           # Intersection
print(set1 - set2)           # Difference
print(set1 ^ set2)           # Symmetric difference
```

### Control Flow
```python
# If statements
age = 18
if age >= 18:
    print("Adult")
elif age >= 13:
    print("Teenager")
else:
    print("Child")

# Ternary operator
status = "Adult" if age >= 18 else "Minor"

# For loops
for i in range(5):           # 0 to 4
    print(i)

for i in range(2, 10, 2):    # Start, stop, step
    print(i)

for fruit in ['apple', 'banana']:
    print(fruit)

for i, fruit in enumerate(['apple', 'banana']):
    print(f"{i}: {fruit}")

# While loops
count = 0
while count < 5:
    print(count)
    count += 1

# Loop control
for i in range(10):
    if i == 3:
        continue  # Skip this iteration
    if i == 7:
        break    # Exit loop
    print(i)
```

### Functions
```python
# Basic function
def greet(name):
    return f"Hello, {name}!"

# Function with default parameters
def greet(name, greeting="Hello"):
    return f"{greeting}, {name}!"

# Function with multiple parameters
def calculate(a, b, operation="add"):
    if operation == "add":
        return a + b
    elif operation == "multiply":
        return a * b
    return None

# Variable arguments
def sum_all(*args):
    return sum(args)

def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

# Lambda functions
square = lambda x: x**2
add = lambda x, y: x + y
```

### Exception Handling
```python
# Basic try-except
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Cannot divide by zero!")

# Multiple exceptions
try:
    value = int(input("Enter a number: "))
    result = 10 / value
except ValueError:
    print("Invalid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    print("This always executes")

# Raising exceptions
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
```

### File Operations
```python
# Reading files
with open('file.txt', 'r') as file:
    content = file.read()        # Read entire file
    
with open('file.txt', 'r') as file:
    lines = file.readlines()     # Read all lines
    
with open('file.txt', 'r') as file:
    for line in file:            # Read line by line
        print(line.strip())

# Writing files
with open('output.txt', 'w') as file:
    file.write('Hello World\n')
    file.writelines(['Line 1\n', 'Line 2\n'])

# Appending to files
with open('output.txt', 'a') as file:
    file.write('Additional content\n')
```