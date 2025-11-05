import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "movie_scene_macro_photography.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Movie Scene Macro Photography",
    "module_title": "Movie Scene Macro Photography Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/movie_scene_macro_photography",
    "imageUrl": "/static/images/ai-generated-images/miniaturre%20scenes/img_1.JPG",
    "date": "",
    "body": get_markdown_content()
}
