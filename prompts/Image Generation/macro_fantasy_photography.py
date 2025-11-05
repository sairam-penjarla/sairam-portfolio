import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "macro_fantasy_photography.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Macro Fantasy Photography",
    "module_title": "Macro Fantasy Photography Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/macro_fantasy_photography",
    "imageUrl": "/static/images/ai-generated-images/warrior%20bee/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
