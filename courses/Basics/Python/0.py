import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "0 Basics of python.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Basics of Python",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Python Full Course for Beginners [2025] (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  },
  {
    "name": "Python Full Course for Beginners (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
    "type": "video"
  },
  {
    "name": "Python Tutorial for Beginners - Learn Python in 5 Hours (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=eWRfhZUzrAc",
    "type": "video"
  },
  {
    "name": "Complete Python Playlist (Krish Naik)",
    "url": "https://www.youtube.com/playlist?list=PLZoTAELRMXVNUL99R4bDlVYsncUNvwUBB",
    "type": "video"
  },
  {
    "name": "Python for Beginners – Full Course [Programming Tutorial]",
    "url": "https://www.youtube.com/watch?v=eWRfhZUzrAc",
    "type": "video"
  },
  {
    "name": "Python Programming Tutorial (CodeWithHarry)",
    "url": "https://www.youtube.com/watch?v=apGV9Kg7ics",
    "type": "video"
  },
  {
    "name": "Python Programming for Beginners to Experts | Full Course Tutorial",
    "url": "https://www.youtube.com/watch?v=2uCXIbkbDSE",
    "type": "video"
  },
  {
    "name": "Python Programming Tutorial (Telusko)",
    "url": "https://www.youtube.com/watch?v=HkdAHXoRtos",
    "type": "video"
  },
  {
    "name": "Python Tutorial for Beginners (Simplilearn)",
    "url": "https://www.youtube.com/watch?v=8JJ101D3knE",
    "type": "video"
  },
  {
    "name": "Python Tutorial for Beginners (freeCodeCamp)",
    "url": "https://www.youtube.com/watch?v=YYXdXT2l-Gg",
    "type": "video"
  }
]

}


