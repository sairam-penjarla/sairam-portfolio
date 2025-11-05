import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "event_flyer.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Event Flyer",
    "module_title": "Event Flyer Prompt",
    "category": "Flyers",
    "url": "/prompts/Flyers/event_flyer",
    "imageUrl": "/static/images/ai-generated-images/holi_flyer/1.JPG",
    "date": "",
    "body": get_markdown_content()
}
