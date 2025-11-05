import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "mythical_and_divine_beings.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Mythical And Divine Beings",
    "module_title": "Mythical And Divine Beings Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/mythical_and_divine_beings",
    "imageUrl": "/static/images/ai-generated-images/ganesh/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
