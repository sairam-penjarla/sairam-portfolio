import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "Module 2.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "category": "Data Science",
    "course_title": "Machine Learning",
    "module_title": "Logistic Regression",
    "url": "/learn/courses/data_science/machine_learning",
    "imageUrl": "https://iamsairamstrprd.blob.core.windows.net/static/images/thumbnails/img_15.jpg",
    "date": "Duration: 4 Weeks",
    "body": get_markdown_content(),
    "resources": [
  {
    "name": "Logistic Regression (Playlist) - StatQuest with Josh Starmer",
    "url": "https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe",
    "type": "video"
  },
  {
    "name": "StatQuest: Logistic Regression (StatQuest with Josh Starmer)",
    "url": "https://www.youtube.com/watch?v=yIYKR4sgzI8",
    "type": "video"
  },
  {
    "name": "Logistic Regression in 3 Minutes (3-Minute Data Science)",
    "url": "https://www.youtube.com/watch?v=EKm0spFxFG4",
    "type": "video"
  },
  {
    "name": "Linear Regression vs Logistic Regression - What's The Difference? (The Comparison Channel)",
    "url": "https://www.youtube.com/watch?v=06en5XqdPkI",
    "type": "video"
  },
  {
    "name": "Logistic Regression (and why it's different from Linear Regression) - Visually Explained",
    "url": "https://www.youtube.com/watch?v=3bvM3NyMiE0",
    "type": "video"
  },
  {
    "name": "Machine Learning Crash Course: Logistic Regression (Google for Developers)",
    "url": "https://www.youtube.com/watch?v=72AHKztZN44",
    "type": "video"
  }
]

}