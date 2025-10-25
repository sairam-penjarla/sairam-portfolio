import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "6. How to build a RAG Backend.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Generative AI",
    "course_title": "Retrieval Augmented Generation",
    "module_title": "How to build a RAG Backend",
    "url": "/learn/courses/generative_ai/rag",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_21.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
