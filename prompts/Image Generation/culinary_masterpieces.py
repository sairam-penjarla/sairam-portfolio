import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "culinary_masterpieces.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Culinary Masterpieces",
    "module_title": "Culinary Masterpieces Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/culinary_masterpieces",
    "imageUrl": "/static/images/ai-generated-images/food%20photography/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
