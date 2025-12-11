import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "8. Understanding Hallucinations in AI.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "AI Basics",
    "module_title": "Understanding Hallucinations in AI",
    "url": "/learn/courses/generative_ai/ai_basics",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_23.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
