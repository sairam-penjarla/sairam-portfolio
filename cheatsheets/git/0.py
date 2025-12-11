import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Git.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Git",
    "module_title": "Git Cheatsheet",
    "category": "Git",
    "url": "/learn/courses/git/Git",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_51.JPG",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
