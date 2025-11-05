import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "horror_story.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Horror Story",
    "module_title": "Horror Story Prompt",
    "category": "Novel Writing",
    "url": "/prompts/Novel Writing/horror_story",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_35.jpg",
    "date": "",
    "body": get_markdown_content()
}
