import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "chocklate_grapes.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Chocklate Grapes",
    "module_title": "Chocklate Grapes Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/chocklate_grapes",
    "imageUrl": "/static/images/ai-generated-images/chocklate%20grapes/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
