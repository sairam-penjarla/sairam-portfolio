import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "data_preprocessing.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Data Preprocessing",
    "module_title": "Basics of Data Preprocessing",
    "category": "Basics",
    "url": "/learn/courses/basics/data_preprocessing",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_7.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
