import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "1 basics.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "SQL",
    "module_title": "Basics of SQL",
    "category": "Basics",
    "url": "/learn/courses/basics/sql",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_10.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "SQL Course for Beginners [Full Course] (Programming with Mosh)",
    "url": "https://www.youtube.com/watch?v=7S_tz1z_5bA",
    "type": "video"
  },
  {
    "name": "SQL Full Course for Beginners (30 Hours) – From Zero to Hero",
    "url": "https://www.youtube.com/watch?v=SSKVgrwhzus",
    "type": "video"
  },
  {
    "name": "SQL Tutorial for Beginners - YouTube (Kevin Stratvert)",
    "url": "https://www.youtube.com/watch?v=h0nxCDiD-zg",
    "type": "video"
  },
  {
    "name": "Learn SQL Beginner to Advanced in Under 4 Hours (Alex The Analyst)",
    "url": "https://www.youtube.com/watch?v=OT1RErkfLNQ",
    "type": "video"
  },
  {
    "name": "SQLZoo: SQL Tutorial (Interactive Practice)",
    "url": "https://sqlzoo.net/wiki/SQL_Tutorial",
    "type": "web"
  }
]
}