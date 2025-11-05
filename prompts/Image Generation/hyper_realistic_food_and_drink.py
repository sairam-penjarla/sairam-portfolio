import os

def get_markdown_content():
    """Reads and returns the latest content of the markdown file"""
    md_path = os.path.join(os.path.dirname(__file__), "hyper_realistic_food_and_drink.md")
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()

course_data = {
    "course_title": "Hyper-realistic Food And Drink",
    "module_title": "Hyper-realistic Food And Drink Prompt",
    "category": "Image Generation",
    "url": "/prompts/Image Generation/hyper_realistic_food_and_drink",
    "imageUrl": "https://iamsairamsa.blob.core.windows.net/static/images/thumbnails/img_11.jpg",
    "date": "",
    "body": get_markdown_content()
}
