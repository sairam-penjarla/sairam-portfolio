import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "langgraph part 3 decision making.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "LangGraph",
    "module_title": "Decision-Making with Routers",
    "url": "/learn/courses/generative_ai/langgraph",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_24.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
