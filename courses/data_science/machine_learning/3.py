import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Module 3.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Machine Learning",
    "module_title": "Decision Trees",
    "url": "/learn/courses/data_science/machine_learning",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_15.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Decision and Classification Trees, Clearly Explained!!! (StatQuest with Josh Starmer)",
    "url": "https://www.youtube.com/watch?v=_L39rN6gz7Y",
    "type": "video"
  },
  {
    "name": "Let’s Write a Decision Tree Classifier from Scratch - Machine Learning Recipes #8 (Google for Developers)",
    "url": "https://www.youtube.com/watch?v=LDRbO9a6XPU",
    "type": "video"
  },
  {
    "name": "Decision Trees - Visually Explained",
    "url": "https://www.youtube.com/watch?v=LLBGiAAZqAM",
    "type": "video"
  },
  {
    "name": "Regression Trees, Clearly Explained!!! (StatQuest with Josh Starmer)",
    "url": "https://www.youtube.com/watch?v=g9c66TUylZ4",
    "type": "video"
  },
  {
    "name": "How to implement Decision Trees from scratch with Python (AssemblyAI)",
    "url": "https://www.youtube.com/watch?v=NxEHSAfFlK8",
    "type": "video"
  },
  {
    "name": "Machine Learning Tutorial Python - 9 Decision Tree (codebasics)",
    "url": "https://www.youtube.com/watch?v=PHxYNGo8NcI",
    "type": "video"
  }
]

}