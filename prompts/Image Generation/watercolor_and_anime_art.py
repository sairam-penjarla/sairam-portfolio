import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "watercolor_and_anime_art.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Watercolor And Anime Art",
    "module_title": "Watercolor And Anime Art Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/watercolor_and_anime_art",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_11.jpg",
    "date": "",
    "body": get_markdown_content()
}
