import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "2 Advance python.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Advance Python",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Advanced Python: Decorators In-Depth (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=zA53Lf-fqcQ",
    "type": "video"
  },
  {
    "name": "Advanced Python: Exception Handling Explained (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=06HauGzxc9s",
    "type": "video"
  },
  {
    "name": "Advanced Python: Logging Tutorial (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=8JJ101D3knE",
    "type": "video"
  },
  {
    "name": "Advanced Python: Context Managers and Generators (freeCodeCamp)",
    "url": "https://www.youtube.com/watch?v=HGOBQPFzWKo",
    "type": "video"
  },
  {
    "name": "Advanced Python: Threading vs Multiprocessing (freeCodeCamp)",
    "url": "https://www.youtube.com/watch?v=8JJ101D3knE",
    "type": "video"
  },
  {
    "name": "Advanced Python: Creating a Programming Language (freeCodeCamp)",
    "url": "https://www.youtube.com/watch?v=1WpKsY9LBlY",
    "type": "video"
  },
  {
    "name": "Advanced Python: Build a Music Recommendation System (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  },
  {
    "name": "Advanced Python: Build a Grocery Store Website (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  },
  {
    "name": "Advanced Python: Build a Data Automation Tool (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  },
  {
    "name": "Advanced Python: Build a Music Recommendation System (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  }
]

}