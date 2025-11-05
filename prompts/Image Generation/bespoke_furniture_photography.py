import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "bespoke_furniture_photography.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Bespoke Furniture Photography",
    "module_title": "Bespoke Furniture Photography Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/bespoke_furniture_photography",
    "imageUrl": "/static/images/ai-generated-images/heart%20sofa/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
