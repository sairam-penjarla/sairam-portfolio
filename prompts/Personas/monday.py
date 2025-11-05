import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "monday.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Monday",
    "module_title": "Monday Prompt",
    "category": "Personas",
    "url": "/prompts/Personas/monday",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_36.jpg",
    "date": "",
    "body": get_markdown_content()
}
