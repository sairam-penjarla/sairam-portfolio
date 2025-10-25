import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "python.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Python",
    "module_title": "Python Interview Questions",
    "category": "Interview Questions",
    "url": "/learn/courses/interview_questions/python",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_41.JPG",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
