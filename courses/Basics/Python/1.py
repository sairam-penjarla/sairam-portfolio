import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "1 Intermediate python.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Intermediate Python",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Intermediate Python Programming Course",
    "url": "https://www.youtube.com/watch?v=HGOBQPFzWKo",
    "type": "video"
  },
  {
    "name": "Intermediate Python Programming Tutorial (2023)",
    "url": "https://www.youtube.com/watch?v=AXkEIvbkmlg",
    "type": "video"
  },
  {
    "name": "Intermediate Python Programming",
    "url": "https://www.youtube.com/playlist?list=PLzMcBGfZo4-nhWva-6OVh1yKWHBs4o_tv",
    "type": "video"
  },
  {
    "name": "Python Full Course for Beginners [2025]",
    "url": "https://www.youtube.com/watch?v=K5KVEU3aaeQ",
    "type": "video"
  },
  {
    "name": "Complete Road Map To Be Expert In Python",
    "url": "https://www.youtube.com/watch?v=bPrmA1SEN2k",
    "type": "video"
  },
  {
    "name": "Intermediate Python Programming",
    "url": "https://www.youtube.com/playlist?list=PL30AETbxgR-cbPtjzN9Sz4WIcl1oBUCcC",
    "type": "video"
  },
  {
    "name": "Intermediate Python - Functions, Modules, Classes and Exceptions",
    "url": "https://www.youtube.com/watch?v=xWsJ_qk-eJE",
    "type": "video"
  },
  {
    "name": "Intermediate Python Tutorials",
    "url": "https://www.youtube.com/playlist?list=PLzMcBGfZo4-nhWva-6OVh1yKWHBs4o_tv",
    "type": "video"
  },
  {
    "name": "Python Full Course for Beginners",
    "url": "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
    "type": "video"
  },
  {
    "name": "Python Programming Tutorial",
    "url": "https://www.youtube.com/watch?v=apGV9Kg7ics",
    "type": "video"
  }
]

}