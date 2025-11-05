import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "2 intermediate.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "SQL",
    "module_title": "Intermediate SQL",
    "category": "Basics",
    "url": "/learn/courses/basics/sql",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_10.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Learn SQL Beginner to Advanced in Under 4 Hours (Alex The Analyst)",
    "url": "https://www.youtube.com/watch?v=OT1RErkfLNQ",
    "type": "video"
  },
  {
    "name": "SQL for Data Analytics – Intermediate Course + Project (Data Analytics)",
    "url": "https://www.youtube.com/watch?v=QKIGsShyEsQ",
    "type": "video"
  },
  {
    "name": "SQL Tutorial: Intermediate SQL | Intro (DataCamp)",
    "url": "https://www.youtube.com/watch?v=FT9SQVsbmGE",
    "type": "video"
  },
  {
    "name": "Intermediate SQL Tutorial | Inner/Outer Joins | Use Cases (Alex The Analyst)",
    "url": "https://www.youtube.com/watch?v=9URM1_2S0ho",
    "type": "video"
  },
  {
    "name": "Practice Intermediate SQL Problems Full Tutorial",
    "url": "https://www.youtube.com/watch?v=ONJtvAH-4LM",
    "type": "video"
  },
  {
    "name": "SQLZoo: SQL Tutorial (Advanced Lessons)",
    "url": "https://sqlzoo.net/wiki/SQL_Tutorial",
    "type": "web"
  }
]
}