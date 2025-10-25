import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Advanced.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Advanced",
    "module_title": "Python Advanced Cheatsheet",
    "category": "Python",
    "url": "/learn/courses/python/Advanced",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_50.JPG",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
