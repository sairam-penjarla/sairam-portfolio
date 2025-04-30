> Flask is a lightweight web framework for Python that is easy to use and great for building web applications, including APIs. In this blog, we’ll walk through the basics of Flask and show you how to get started with creating your own simple API. By the end of this guide, you’ll have a working Flask API and the skills to build more advanced web applications.
> 

# Table of contents

## **Module 1: Setting Up Flask**

To get started with Flask, you'll first need to install the framework.

### **Step 1: Install Flask**

You can install Flask using `pip`, Python’s package manager. Open your terminal or command prompt and run the following command:

```bash
pip install Flask

```

### **Step 2: Create Your First Flask App**

Now that Flask is installed, let's write a basic Flask application. This will be a simple “Hello, World!” app.

### **Code Example 1: Basic Flask Application**

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(debug=True)

```

### **Explanation:**

1. **Import Flask**: We import the `Flask` class from the `flask` package.
2. **Create an App**: We create an instance of the Flask class. The `__name__` argument tells Flask where to look for the app’s resources.
3. **Define a Route**: The `@app.route('/')` decorator tells Flask that the `hello_world()` function will handle requests to the root URL (`/`).
4. **Run the App**: The `app.run(debug=True)` line runs the Flask application in development mode, which allows automatic reloading of the app and provides debugging information in case of errors.

### **Output:**

When you run this script, open your browser and navigate to `http://127.0.0.1:5000/`. You should see the message **"Hello, World!"** displayed in your browser.

---

## **Module 2: Understanding Flask Routes**

In Flask, routes are used to define URL patterns that map to specific functions in your app. Let’s explore more about routes and how you can use them to handle different requests.

### **Step 1: Multiple Routes**

You can define different routes to handle different parts of your web application.

### **Code Example 2: Multiple Routes**

```python
from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    return 'Welcome to the Home Page!'

@app.route('/about')
def about():
    return 'This is the About Page.'

@app.route('/contact')
def contact():
    return 'This is the Contact Page.'

if __name__ == '__main__':
    app.run(debug=True)

```

### **Explanation:**

1. We define three routes: `/`, `/about`, and `/contact`.
2. Each route is mapped to a specific function: `home()`, `about()`, and `contact()`.
3. When you visit `http://127.0.0.1:5000/`, `http://127.0.0.1:5000/about`, or `http://127.0.0.1:5000/contact`, you’ll see different messages depending on which URL you visit.

### **Output:**

- Visiting `/` will show **"Welcome to the Home Page!"**
- Visiting `/about` will show **"This is the About Page."**
- Visiting `/contact` will show **"This is the Contact Page."**

---

## **Module 3: Handling HTTP Methods**

Flask supports various HTTP methods such as GET, POST, PUT, DELETE, etc. The most common method is `GET`, which retrieves information from the server.

Let’s explore how you can use different HTTP methods in Flask.

### **Step 1: Handling POST Requests**

You can use the `POST` method to send data to the server, for example, when submitting a form.

### **Code Example 3: Handling POST Requests**

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    return f'Username: {username}, Password: {password}'

if __name__ == '__main__':
    app.run(debug=True)

```

### **Explanation:**

1. The `@app.route('/login', methods=['POST'])` decorator specifies that the `login()` function will handle POST requests at the `/login` route.
2. Inside the `login()` function, we use `request.form` to get the data submitted in the POST request (username and password).
3. The `login()` function returns the username and password as a response.

### **Testing POST Request with cURL:**

To test the POST request, you can use **cURL** from the terminal:

```bash
curl -X POST -F "username=myuser" -F "password=mypassword" <http://127.0.0.1:5000/login>

```

### **Output:**

You should see the following output:

```
Username: myuser, Password: mypassword

```

---

### **Step 2: Handling GET Requests**

The `GET` method is used to retrieve data from the server. Let’s see how you can handle a GET request in Flask.

### **Code Example 4: Handling GET Requests**

```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/greet', methods=['GET'])
def greet():
    name = request.args.get('name', 'Guest')
    return f'Hello, {name}!'

if __name__ == '__main__':
    app.run(debug=True)

```

### **Explanation:**

1. The `@app.route('/greet', methods=['GET'])` decorator specifies that the `greet()` function will handle GET requests at the `/greet` route.
2. We use `request.args.get('name', 'Guest')` to retrieve the `name` parameter from the URL query string (e.g., `/greet?name=Alice`). If no name is provided, it defaults to `'Guest'`.
3. The `greet()` function returns a greeting message.

### **Testing GET Request:**

You can test this with a web browser or cURL:

```bash
curl "<http://127.0.0.1:5000/greet?name=Alice>"

```

### **Output:**

The output will be:

```
Hello, Alice!

```

If you don’t provide a name:

```bash
curl "<http://127.0.0.1:5000/greet>"

```

You’ll see:

```
Hello, Guest!

```

---

## **Module 4: Flask API with JSON**

Flask also allows you to easily return data in JSON format, which is essential for building APIs.

### **Step 1: Returning JSON Responses**

You can return JSON responses using Flask’s `jsonify` function.

### **Code Example 5: Returning JSON Data**

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api')
def api():
    data = {
        'name': 'Alice',
        'age': 30
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)

```

### **Explanation:**

1. We import the `jsonify` function from Flask.
2. The `api()` function returns a dictionary containing some data (name and age).
3. The `jsonify()` function converts the dictionary into a JSON response.

### **Output:**

When you visit `http://127.0.0.1:5000/api`, the response will be:

```json
{
  "name": "Alice",
  "age": 30
}

```

---

## **Conclusion**

In this guide, we’ve learned the basics of Flask and how to create a simple API. Here’s a quick summary of what we’ve covered:

- **Setting up Flask**: Installing Flask and creating your first Flask app.
- **Routes**: Understanding how to create and handle different routes.
- **HTTP Methods**: Handling `GET` and `POST` requests.
- **JSON Responses**: Returning data in JSON format, which is essential for APIs.

### **Key Takeaways**

- Always practice hands-on by creating your own small Flask projects.
- Use **PyCharm** or **VSCode** for easier development and debugging.
- Test your endpoints using tools like cURL or Postman.

I encourage you to build on this knowledge and try creating more complex APIs with Flask. Flask makes it easy to create simple yet powerful web applications and APIs, and with practice, you’ll get more comfortable building and scaling your projects.

Happy coding!