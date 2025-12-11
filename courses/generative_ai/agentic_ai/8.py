import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "8. Building an Entire Agentic AI Backend: From Components to Production System.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "Agentic AI",
    "module_title": "Building an Entire Agentic AI Backend",
    "url": "/learn/courses/generative_ai/agentic_ai",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_22.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
