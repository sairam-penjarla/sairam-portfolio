import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "opencv.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Computer Vision",
    "module_title": "Basics of computer vision",
    "url": "/learn/courses/data_science/computer_vision",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_13.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
  

}