import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "charecter_design.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Charecter Design",
    "module_title": "Charecter Design Prompt",
    "category": "Novel Writing",
    "url": "/prompts/Novel Writing/charecter_design",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_34.jpg",
    "date": "",
    "body": get_markdown_content()
}
