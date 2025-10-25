import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Huggingface.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Deep Learning",
    "module_title": "Basics of HuggingFace",
    "url": "/learn/courses/data_science/deep_learning",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_14.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}