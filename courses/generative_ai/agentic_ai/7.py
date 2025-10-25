import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "7. Building Multiple Agents: Specialization Through Agent Teams.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "Agentic AI",
    "module_title": "Building Multiple Agents",
    "url": "/learn/courses/generative_ai/agentic_ai",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_22.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
