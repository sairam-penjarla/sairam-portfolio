import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "fanciful_animal_hybrids.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Fanciful Animal Hybrids",
    "module_title": "Fanciful Animal Hybrids Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/fanciful_animal_hybrids",
    "imageUrl": "/static/images/ai-generated-images/glass%20peacock/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
