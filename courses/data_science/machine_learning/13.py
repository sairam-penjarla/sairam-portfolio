import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Module 13.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Machine Learning",
    "module_title": "Natural Language Processing (NLP) - Text Preprocessing",
    "url": "/learn/courses/data_science/machine_learning",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_15.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
