import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "SSH.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "SSH",
    "module_title": "SSH Cheatsheet",
    "category": "Command Line Interface",
    "url": "/learn/courses/command_line_interface/SSH",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_12.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
