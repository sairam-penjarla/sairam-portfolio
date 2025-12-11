import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "git.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Git",
    "module_title": "Basics of Git",
    "category": "Basics",
    "url": "/learn/courses/basics/git",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_8.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources" : [
  {
    "name": "Git and GitHub for Beginners - Crash Course",
    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "type": "video"
  },
  {
    "name": "Git Tutorial for Beginners: Learn Git in 1 Hour",
    "url": "https://www.youtube.com/watch?v=8JJ101D3knE",
    "type": "video"
  },
  {
    "name": "Git and GitHub for Beginners - Full Course",
    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "type": "video"
  },
  {
    "name": "Git Tutorial for Dummies",
    "url": "https://www.youtube.com/watch?v=mJ-qvsxPHpY",
    "type": "video"
  },
  {
    "name": "Git and GitHub Tutorial for Beginners",
    "url": "https://www.youtube.com/watch?v=tRZGeaHPoaw",
    "type": "video"
  },
  {
    "name": "Git and GitHub for Beginners",
    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "type": "video"
  },
  {
    "name": "Git Tutorial for Complete Beginners",
    "url": "https://www.youtube.com/watch?v=kY5HtrkjSj0",
    "type": "video"
  },
  {
    "name": "Git and GitHub for Beginners",
    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "type": "video"
  },
  {
    "name": "Git Tutorial for Dummies",
    "url": "https://www.youtube.com/watch?v=mJ-qvsxPHpY",
    "type": "video"
  },
  {
    "name": "Git and GitHub for Beginners",
    "url": "https://www.youtube.com/watch?v=RGOj5yH7evk",
    "type": "video"
  }
]


}