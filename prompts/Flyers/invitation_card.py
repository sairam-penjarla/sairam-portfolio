import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "invitation_card.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Invitation Card",
    "module_title": "Invitation Card Prompt",
    "category": "Flyers",
    "url": "/prompts/Flyers/invitation_card",
    "imageUrl": "/static/images/ai-generated-images/invitation%20card/1.PNG",
    "date": "",
    "body": get_markdown_content()
}
