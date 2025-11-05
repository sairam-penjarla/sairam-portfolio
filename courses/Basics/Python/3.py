import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "3 Python Flask Library.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Flask Framework in python",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Flask Official Tutorial",
    "url": "https://flask.palletsprojects.com/en/stable/tutorial/",
    "type": "web"
  },
  {
    "name": "The Flask Mega-Tutorial by Miguel Grinberg",
    "url": "https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world",
    "type": "web"
  },
  {
    "name": "GeeksforGeeks Flask Tutorial",
    "url": "https://www.geeksforgeeks.org/python/flask-tutorial/",
    "type": "web"
  },
  {
    "name": "DigitalOcean Flask Blog Tutorial",
    "url": "https://www.digitalocean.com/community/tutorials/how-to-make-a-web-application-using-flask-in-python-3",
    "type": "web"
  },
  {
    "name": "Codecademy Learn Flask",
    "url": "https://www.codecademy.com/learn/learn-flask",
    "type": "web"
  },
  {
    "name": "Python Flask Tutorial for Beginners (Edureka)",
    "url": "https://www.youtube.com/watch?v=lj4I_CvBnt0",
    "type": "video"
  },
  {
    "name": "Flask Tutorial in Visual Studio Code",
    "url": "https://code.visualstudio.com/docs/python/tutorial-flask",
    "type": "web"
  },
  {
    "name": "Python Website Full Tutorial - Flask, Authentication, Databases & More",
    "url": "https://www.youtube.com/watch?v=dam0GPOAvVI",
    "type": "video"
  },
  {
    "name": "Flask Tutorial - GeeksforGeeks",
    "url": "https://www.geeksforgeeks.org/python/flask-tutorial/",
    "type": "web"
  },
  {
    "name": "Learn Python Flask Tutorial - A Web Framework for Python - DataFlair",
    "url": "https://data-flair.training/blogs/python-flask-tutorial/",
    "type": "web"
  }
]

}