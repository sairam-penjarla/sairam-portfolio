import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "langgraph part 9 web api integration.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "LangGraph",
    "module_title": "Integrating Web and API Data for Enhanced Context",
    "url": "/learn/courses/generative_ai/langgraph",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_24.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
