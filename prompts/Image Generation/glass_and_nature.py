import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "glass_and_nature.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Glass And Nature",
    "module_title": "Glass And Nature Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/glass_and_nature",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_11.jpg",
    "date": "",
    "body": get_markdown_content()
}
