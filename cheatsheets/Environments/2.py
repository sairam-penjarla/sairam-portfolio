import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Anaconda.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Anaconda",
    "module_title": "Anaconda Cheatsheet",
    "category": "Environments",
    "url": "/learn/courses/environments/Anaconda",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_18.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
