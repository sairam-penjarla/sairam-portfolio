import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "sunny.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Sunny",
    "module_title": "Sunny Prompt",
    "category": "Personas",
    "url": "/prompts/Personas/sunny",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_63.JPG",
    "date": "",
    "body": get_markdown_content()
}
