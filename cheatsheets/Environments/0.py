import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "PIP.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "pip",
    "module_title": "pip Cheatsheet",
    "category": "Environments",
    "url": "/learn/courses/environments/PIP",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_16.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
