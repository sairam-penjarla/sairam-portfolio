import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Module 1.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Machine Learning",
    "module_title": "Linear Regression",
    "url": "/learn/courses/data_science/machine_learning",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_15.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Linear Regression, Clearly Explained!!! (StatQuest with Josh Starmer)",
    "url": "https://www.youtube.com/watch?v=7ArmBVF2dCs",
    "type": "video"
  },
  {
    "name": "Machine Learning Tutorial Python - 2: Linear Regression Single Variable (codebasics)",
    "url": "https://www.youtube.com/watch?v=8jazNUpO3lQ",
    "type": "video"
  },
  {
    "name": "Linear Regression Indepth Maths Intuition - Data Science (Krish Naik)",
    "url": "https://www.youtube.com/watch?v=1-OGRohmH2s",
    "type": "video"
  },
  {
    "name": "Linear Regression in 3 Minutes (3-Minute Data Science)",
    "url": "https://www.youtube.com/watch?v=3dhcmeOTZ_Q&pp=ygUSbGluZWFyIHJlZ3Jlc3Npb24g",
    "type": "video"
  }
]

}