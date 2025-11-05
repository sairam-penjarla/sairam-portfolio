import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "handmade_nature.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Handmade Nature",
    "module_title": "Handmade Nature Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/handmade_nature",
    "imageUrl": "/static/images/ai-generated-images/knitted%20butterfly/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
