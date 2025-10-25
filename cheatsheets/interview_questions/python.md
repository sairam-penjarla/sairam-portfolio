## Easy

### **Basics & Syntax**

1. **What are Python’s key features?**

   * Interpreted, high-level, dynamically typed, garbage collected, multi-paradigm (procedural, OOP, functional), large standard library, readable syntax.

2. **What is Python’s difference from C++/Java?**

   * Python is interpreted (no compilation), dynamically typed, memory-managed automatically, with simpler syntax. C++/Java are compiled and statically typed.

3. **How do you declare a variable in Python?**

   ```python
   x = 10
   name = "Alice"
   ```

4. **What is the difference between `==` and `is`?**

   * `==` checks value equality.
   * `is` checks object identity (whether two variables point to the same object).

5. **What are Python’s data types?**

   * Numeric: int, float, complex
   * Sequence: list, tuple, range
   * Text: str
   * Set types: set, frozenset
   * Mapping: dict
   * Boolean: bool

6. **How do you convert a string to an integer?**

   ```python
   num = int("123")
   ```

7. **Difference between list, tuple, and set?**

   * **List**: ordered, mutable, allows duplicates.
   * **Tuple**: ordered, immutable, allows duplicates.
   * **Set**: unordered, mutable, no duplicates.

8. **How do you check the type of a variable?**

   ```python
   type(x)
   ```

9. **Explain Python’s memory management.**

   * Uses reference counting and garbage collection (via `gc` module) to manage memory automatically.

10. **What are mutable and immutable types?**

* **Mutable:** list, dict, set (can change values).
* **Immutable:** int, float, str, tuple (cannot change values).

---

### **Operators & Expressions**

11. **Difference between `/` and `//`**

* `/` → float division.
* `//` → floor (integer) division.

12. **Difference between `+=` and `=`**

* `=` assigns a new value.
* `+=` adds to existing value.

13. **Explain membership operators `in` and `not in`**

* `in` → checks if element exists.
* `not in` → checks if element does not exist.

14. **How do you swap two variables?**

```python
a, b = b, a
```

15. **Difference between `and`, `or`, `not`**

* `and` → True if both True
* `or` → True if any True
* `not` → negates value

---

### **Control Flow**

16. **How do `if`, `elif`, and `else` work?**

* Conditional branching. Example:

```python
if x > 0:
    print("Positive")
elif x == 0:
    print("Zero")
else:
    print("Negative")
```

17. **`for` vs `while` loops**

* `for` → iterates over sequences.
* `while` → loops until condition is False.

18. **Explain `break` and `continue`**

* `break` → exits loop.
* `continue` → skips current iteration.

19. **What is the `pass` statement?**

* Placeholder for empty code block, does nothing.

```python
if True:
    pass
```

20. **How do you iterate over a dictionary?**

```python
for key, value in my_dict.items():
    print(key, value)
```

---

### **Functions**

21. **How do you define a function?**

```python
def greet(name):
    return f"Hello, {name}"
```

22. **What are *args and **kwargs?**

* `*args` → tuple of positional arguments.
* `**kwargs` → dictionary of keyword arguments.

23. **Explain default and keyword arguments**

```python
def greet(name="Alice"):
    print(f"Hello {name}")
greet()  # Hello Alice
```

24. **What is a lambda function?**

```python
square = lambda x: x**2
print(square(5))  # 25
```

25. **Difference between return and print**

* `print` → displays output.
* `return` → gives value back to caller.

26. **How do you pass a list to a function?**

```python
def process(lst):
    return [x*2 for x in lst]
```

27. **What are Python decorators?**

* Functions that modify behavior of other functions.

```python
def decorator(func):
    def wrapper():
        print("Before")
        func()
        print("After")
    return wrapper
```

28. **Explain nested functions**

* Functions defined inside other functions. Can access outer variables.

29. **What is recursion?**

* Function calling itself until a base condition.

30. **Example of recursion**

```python
def factorial(n):
    if n == 0:
        return 1
    return n * factorial(n-1)
```

---

### **Data Structures**

31. **How to create list, tuple, set, dict**

```python
lst = [1,2,3]
tup = (1,2,3)
st = {1,2,3}
d = {"a":1, "b":2}
```

32. **Difference between `append` and `extend`**

* `append` → adds one element.
* `extend` → adds multiple elements from iterable.

33. **Remove duplicates from a list**

```python
lst = [1,2,2,3]
lst = list(set(lst))
```

34. **Merge two dictionaries**

```python
d1 = {'a':1}; d2 = {'b':2}
d1.update(d2)
```

35. **Shallow vs deep copy**

* Shallow copy → copies references, changes reflect in original.
* Deep copy → copies objects recursively.

36. **Explain list slicing**

```python
lst = [1,2,3,4]
lst[1:3]  # [2,3]
lst[::-1] # reverse
```

37. **Difference between stack and queue**

* Stack → LIFO
* Queue → FIFO

38. **Implement a stack**

```python
stack = []
stack.append(1)
stack.pop()
```

39. **Implement a queue**

```python
from collections import deque
queue = deque([1,2])
queue.append(3)
queue.popleft()
```

40. **Sort a list**

```python
lst = [3,1,2]
lst.sort()  # ascending
lst.sort(reverse=True)  # descending
```

## **Medium (41–75) – Questions with Answers**

### **OOP Concepts**

41. **What is object-oriented programming (OOP) in Python?**

* OOP is a paradigm using **classes and objects** to model real-world entities, supporting **encapsulation, inheritance, polymorphism, and abstraction**.

42. **Difference between class and object**

* **Class** → blueprint/template.
* **Object** → instance of a class.

43. **Explain instance, class, and static methods**

```python
class MyClass:
    def instance_method(self):  # needs object
        pass
    @classmethod
    def class_method(cls):     # uses class
        pass
    @staticmethod
    def static_method():       # independent
        pass
```

44. **What is inheritance?**

* A class can inherit attributes/methods from another class.

45. **Single, multiple, multilevel inheritance**

* **Single**: Child inherits one parent.
* **Multiple**: Child inherits multiple parents.
* **Multilevel**: Inheritance chain (grandparent → parent → child).

46. **What is polymorphism?**

* Ability of an object to take multiple forms, e.g., method overriding, operator overloading.

47. **Encapsulation and abstraction**

* **Encapsulation** → restricts access using private/protected variables.
* **Abstraction** → hides implementation details using abstract classes or methods.

48. **Difference between `__str__` and `__repr__`**

* `__str__` → readable representation for users.
* `__repr__` → unambiguous representation for developers.

49. **Explain `super()` in Python**

* Used to call a parent class method or constructor in inheritance.

50. **Python special (dunder) methods**

* Methods with double underscores, e.g., `__init__`, `__str__`, `__add__`, used for operator overloading, object lifecycle, etc.

---

### **Modules & Libraries**

51. **How do you import a module?**

```python
import math
from math import sqrt
```

52. **Difference: `import module` vs `from module import function`**

* `import module` → accesses function via `module.func()`
* `from module import func` → access directly as `func()`

53. **What is Python’s `os` module used for?**

* Interact with OS: file operations, directories, environment variables.

54. **How to read/write a file**

```python
# Read
with open("file.txt") as f:
    data = f.read()
# Write
with open("file.txt", "w") as f:
    f.write("Hello")
```

55. **Difference between text and binary files**

* Text → readable strings (`"r"`, `"w"`)
* Binary → bytes (`"rb"`, `"wb"`)

56. **Exception handling**

```python
try:
    x = 1/0
except ZeroDivisionError:
    print("Error")
```

57. **Explain `try`, `except`, `finally`**

* `try` → code block to test
* `except` → handle exceptions
* `finally` → runs always, e.g., cleanup

58. **What are Python packages?**

* Collection of modules organized in directories with `__init__.py`.

59. **Difference between `sys` and `os` modules**

* `sys` → Python interpreter-specific info (args, path).
* `os` → OS-level operations.

60. **How to install external libraries**

```bash
pip install package_name
```

---

### **Python Internals**

61. **Explain Python’s Global Interpreter Lock (GIL)**

* Ensures **only one thread executes Python bytecode at a time**, even in multi-threading.

62. **Difference between Python 2 and 3**

* Python 3 → Unicode by default, print() function, division returns float.
* Python 2 → ASCII strings by default, print statement, integer division.

63. **Python’s garbage collection**

* Uses **reference counting** + cyclic GC (`gc` module) to free memory.

64. **Python iterators and generators**

* **Iterator** → object that can be iterated (`__iter__` + `__next__`)
* **Generator** → function that yields values lazily using `yield`.

65. **Difference between `yield` and `return`**

* `return` → exits function with value.
* `yield` → pauses function and returns generator.

66. **What is the `with` statement?**

* Context manager; automatically closes resources.

```python
with open("file.txt") as f:
    data = f.read()
```

67. **Explain `zip()` function**

* Combines iterables element-wise:

```python
a=[1,2]; b=[3,4]
list(zip(a,b))  # [(1,3),(2,4)]
```

68. **How does Python manage memory?**

* Reference counting, garbage collection, private heap for objects.

69. **Deep copy vs shallow copy**

* Shallow → copies reference
* Deep → copies objects recursively

```python
import copy
copy.deepcopy(obj)
```

70. **Explain `enumerate()` function**

```python
lst = ['a','b']
for idx, val in enumerate(lst):
    print(idx, val)
```

---

### **Data Handling**

71. **Difference: list, deque, queue**

* **List** → basic sequences, O(n) pop(0)
* **Deque** → double-ended queue, O(1) append/pop both ends
* **Queue** → FIFO queue in `queue` module

72. **Merge two lists without duplicates**

```python
lst1 = [1,2]; lst2 = [2,3]
merged = list(set(lst1+lst2))
```

73. **Difference: `map()`, `filter()`, `reduce()`**

* `map(func, iter)` → applies func to all items
* `filter(func, iter)` → selects items where func is True
* `reduce(func, iter)` → cumulative reduction (from `functools`)

74. **Convert list to string**

```python
lst = ['a','b']
s = ''.join(lst)
```

75. **Sort dictionary by values**

```python
d = {'a':3,'b':1}
sorted(d.items(), key=lambda x:x[1])
```

---

## **Hard (76–100) – Questions with Answers**

### **Advanced OOP & Design Patterns**

76. **Explain metaclasses in Python**

* A metaclass is a **class of a class** that defines how a class behaves. It can control class creation, inheritance, and attributes.

```python
class Meta(type):
    def __new__(cls, name, bases, dct):
        print("Creating class", name)
        return super().__new__(cls, name, bases, dct)
class MyClass(metaclass=Meta):
    pass
```

77. **Difference between `__new__` and `__init__`**

* `__new__` → creates a new instance (constructor)
* `__init__` → initializes the instance (initializer)

78. **What are descriptors in Python?**

* Objects defining `__get__`, `__set__`, `__delete__` methods to manage attribute access. Example: `property()` uses descriptors internally.

79. **How does multiple inheritance work?**

* Python uses **Method Resolution Order (MRO)** to determine the order in which base classes are searched.

```python
print(MyClass.__mro__)
```

80. **Explain the singleton pattern in Python**

* Ensures only one instance exists:

```python
class Singleton:
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance
```

81. **How to implement a factory design pattern**

```python
class Dog: pass
class Cat: pass
def get_pet(pet="dog"):
    pets = {"dog": Dog, "cat": Cat}
    return pets[pet]()
```

82. **Difference between composition and inheritance**

* **Inheritance** → "is-a" relationship
* **Composition** → "has-a" relationship

83. **Python’s abstract base classes (ABC)**

* Used to define abstract methods that subclasses must implement:

```python
from abc import ABC, abstractmethod
class Animal(ABC):
    @abstractmethod
    def speak(self):
        pass
```

84. **How to implement operator overloading**

```python
class Point:
    def __init__(self, x, y): self.x, self.y = x, y
    def __add__(self, other): return Point(self.x+other.x, self.y+other.y)
```

85. **Explain the proxy design pattern**

* Provides a **placeholder for another object** to control access or add functionality.

---

### **Advanced Data Structures & Algorithms**

86. **Implement a linked list**

```python
class Node:
    def __init__(self, data): self.data, self.next = data, None
class LinkedList:
    def __init__(self): self.head = None
    def append(self, data):
        new_node = Node(data)
        if not self.head: self.head = new_node
        else:
            curr = self.head
            while curr.next: curr = curr.next
            curr.next = new_node
```

87. **Implement a binary search tree**

```python
class Node:
    def __init__(self, val): self.val, self.left, self.right = val, None, None
```

88. **Difference between BFS and DFS**

* **BFS** → level-order, queue-based
* **DFS** → depth-first, stack/recursion-based

89. **Queue using two stacks**

* Push to stack1, pop from stack2, move elements when needed.

90. **Detect cycle in linked list**

```python
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast: return True
    return False
```

91. **Stack using Python list**

```python
stack = []
stack.append(1)
stack.pop()
```

92. **Reverse a linked list**

```python
prev, curr = None, head
while curr:
    nxt = curr.next
    curr.next = prev
    prev = curr
    curr = nxt
head = prev
```

93. **Find largest/smallest without built-in**

```python
def max_val(lst):
    max_val = lst[0]
    for i in lst: max_val = i if i>max_val else max_val
    return max_val
```

94. **Difference between heapq and priority queue**

* `heapq` → min-heap implemented on list
* `PriorityQueue` → thread-safe, queue-based

95. **Implement LRU cache**

```python
from collections import OrderedDict
class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    def get(self, key):
        if key not in self.cache: return -1
        self.cache.move_to_end(key)
        return self.cache[key]
    def put(self, key, value):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.capacity: self.cache.popitem(last=False)
```

---

### **Concurrency & Performance**

96. **Difference between multithreading and multiprocessing**

* **Threading** → multiple threads, shared memory, limited by GIL
* **Multiprocessing** → separate processes, true parallelism

97. **Thread-safe counter**

```python
from threading import Lock
class Counter:
    def __init__(self): self.count = 0; self.lock = Lock()
    def increment(self):
        with self.lock: self.count += 1
```

98. **Explain async and await**

* Allows asynchronous programming without blocking:

```python
import asyncio
async def foo(): await asyncio.sleep(1)
```

99. **Python’s `concurrent.futures` module**

* Simplifies thread/process pools with `ThreadPoolExecutor` or `ProcessPoolExecutor`.

100. **Profile a Python program**

```python
import cProfile
def my_func(): pass
cProfile.run("my_func()")
```