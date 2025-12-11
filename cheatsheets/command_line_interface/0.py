import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Cli.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "CLI",
    "module_title": "Command Line Interface Cheatsheet",
    "category": "Command Line Interface",
    "url": "/learn/courses/command_line_interface/Cli",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_11.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content()
}
