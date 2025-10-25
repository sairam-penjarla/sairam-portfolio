import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "0 sql.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "SQL",
    "module_title": "Introduction to SQL",
    "category": "Basics",
    "url": "/learn/courses/basics/sql",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_10.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}