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

# **Module 1:**

Let’s start with the fundamentals of Python. These are the building blocks of any Python program.

## **1. Variables**

Variables in Python are used to store data values. You can think of them like containers that hold values.

### **Legal Variable Names**

In Python, variable names should follow these rules:

- They can contain letters (A-Z, a-z), digits (0-9), and underscores (_).
- They must start with a letter or an underscore, not a digit.

Here are some legal and illegal variable names:

**Legal:**

```python
my_var = "John"
_my_var = "John"
MYVAR = "John"
myVar = "John"
myvar2 = "John"

```

**Illegal:**

```python
2myvar = "John"  # Cannot start with a number
my-var = "John"  # Hyphen is not allowed
my var = "John"  # Space is not allowed

```

### **Multiple Assignment**

You can assign multiple values to multiple variables in a single line.

```python
x, y, z = "Orange", "Banana", "Cherry"
print(x)
print(y)
print(z)

```

**Output:**

```
Orange
Banana
Cherry

```

### **Global Variables**

Global variables are variables that are defined outside of any function and can be accessed anywhere in your code.

```python
def myfunc():
    global x
    x = "fantastic"

myfunc()
print("Python is " + x)

```

**Output:**

```
Python is fantastic

```

---

## **2. Data Types**

Python supports several built-in data types, each designed to handle different types of data.

### **Basic Data Types**

- **String (str)**: A sequence of characters.
- **Integer (int)**: Whole numbers.
- **Float (float)**: Numbers with decimals.
- **Boolean (bool)**: True or False.

Here’s how to check the type of a variable:

```python
x = "Hello"
y = 5
z = 3.14
print(type(x))  # <class 'str'>
print(type(y))  # <class 'int'>
print(type(z))  # <class 'float'>

```

### **Uncommon Data Types**

```python
x = 35e3
y = 12E4
z = -87.7e100
print(type(x))  # <class 'float'>
print(type(y))  # <class 'float'>
print(type(z))  # <class 'float'>

```

### **Type Casting**

You can convert between different data types in Python.

```python
x = 1   # int
y = 2.8 # float
z = 1j  # complex

# Convert from int to float:
a = float(x)

# Convert from float to int:
b = int(y)

# Convert from int to complex:
c = complex(x)

print(a, b, c)
print(type(a), type(b), type(c))

```

**Output:**

```
1.0 2 1+0j
<class 'float'> <class 'int'> <class 'complex'>

```

---

## **3. String Operations**

Strings are one of the most commonly used data types in Python. Let’s see how you can manipulate them.

### **Triple Quoted Strings**

Use triple quotes to define multi-line strings.

```python
a = '''Lorem ipsum dolor sit amet,
consectetur adipiscing elit,
sed do eiusmod tempor incididunt
ut labore et dolore magna aliqua.'''
print(a)

```

### **String as Array**

Strings in Python are like arrays – you can access characters by index.

```python
a = "Hello, World!"
print(a[1])  # Output: e

```

### **Looping Through a String**

You can loop through each character in a string.

```python
for x in "banana":
    print(x)

```

### **String Length**

Find out how long a string is.

```python
a = "Hello, World!"
print(len(a))  # Output: 13

```

### **Check if Substring Exists**

Check if a certain substring is present in a string.

```python
txt = "The best things in life are free!"
print("free" in txt)  # Output: True

```

### **String Slicing**

You can slice strings to extract a portion of it.

```python
b = "Hello, World!"
print(b[:5])  # Output: Hello

```

### **String Concatenation**

Combine two strings together.

```python
a = "Hello"
b = "World"
c = a + b
print(c)  # Output: HelloWorld

```

### **String Formatting**

You can insert variables into strings using `.format()`.

```python
age = 36
txt = "My name is John, and I am {}"
print(txt.format(age))  # Output: My name is John, and I am 36

```

### **Escape Characters**

Escape characters allow special formatting like newlines or tabs.

```python
print("Hello\\nWorld!")  # Output: Hello (new line) World!

```

---

## **4. Conditional Statements**

Control the flow of your program using conditional statements.

```python
a = 2
b = 330
if a > b:
    print("A")
else:
    print("B")

```

**Output:**

```
B

```

### **Logical Operators: and, or, not**

```python
# 'and' operator
x = 5
y = 10
if x > 0 and y > 0:
    print("Both x and y are greater than 0.")  # Output: Both x and y are greater than 0.

# 'or' operator
name = "Alice"
age = 25
if name == "Alice" or age == 25:
    print("Either the name is Alice or the age is 25.")  # Output: Either the name is Alice or the age is 25.

# 'not' operator
flag = False
if not flag:
    print("The flag is not True.")  # Output: The flag is not True.

```

### **Conclusion: Practice Makes Perfect!**

Now that you’ve learned the basics of Python, the key to mastering programming is practice. Try solving problems on platforms like [**CodeChef**](https://www.codechef.com/practice-old) and learn about [**Data Structures and Algorithms**](https://www.geeksforgeeks.org/data-structures/) to improve your skills further.

Don't forget, the best way to learn is by doing. Use **Google Colab** to write, test, and run your Python code. You'll find that hands-on practice is the most important step in becoming proficient in Python.

Happy coding, and keep practicing!