import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "4 Project repo structure.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Python Project Repo Structure",
    "category": "Basics",
    "url": "/learn/courses/basics/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_9.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Python Repo Basics",
    "url": "https://youtube.com/playlist?list=PL9wcLF5LwK5BKbEUNtbaCDyob9x0Ktui4&si=UEMpD-kOVLfAm5Z",
    "type": "video"
  }
]

}