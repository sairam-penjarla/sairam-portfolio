from flask import Flask, render_template
from flask import Flask, render_template, abort
import os
import re
from flask import Flask, render_template, jsonify, send_file, abort
import os
import json
from markupsafe import Markup

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home_sections/newhome.html')

@app.route('/test')
def test():
    return render_template('test.html')

# @app.route('/newhome')
# def newhome():
#     return render_template('newhome.html')

@app.route('/resources')
def resources():
    return render_template('resources/resources.html')

@app.route('/blogs', defaults={'path': ''})
@app.route('/blogs/', defaults={'path': ''})
@app.route('/blogs/<path:path>')
def blogs(path):
    def get_folder_structure(root_dir, rel_path=""):
        structure = []
        for item in sorted(os.listdir(root_dir)):
            if item == '.DS_Store':
                continue
            path_full = os.path.join(root_dir, item)
            rel_item_path = os.path.join(rel_path, item)
            if os.path.isdir(path_full):
                structure.append({
                    "type": "folder",
                    "name": item,
                    "path": rel_item_path,
                    "children": get_folder_structure(path_full, rel_item_path)
                })
            else:
                structure.append({
                    "type": "file",
                    "name": item,
                    "path": rel_item_path
                })
        return structure

    blogs_dir = os.path.join(app.root_path, "blogs")
    sidebar_structure = get_folder_structure(blogs_dir)

    # Absolute safe path
    target_path = os.path.abspath(os.path.join(blogs_dir, path))
    if not target_path.startswith(blogs_dir):
        abort(403)  # Prevent directory traversal

    file_content = None
    if os.path.isfile(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

    return render_template(
        'blogs/blogs.html',
        page_prefix = "blogs",
        sidebar=sidebar_structure,
        initial_path=path,
        file_content=file_content
    )


@app.route('/blog/<path:filename>')
def get_blog_file(filename):
    blogs_dir = os.path.join(app.root_path, "blogs")
    file_path = os.path.abspath(os.path.join(blogs_dir, filename))

    if not file_path.startswith(blogs_dir):
        abort(403)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    else:
        abort(404)

        
@app.route('/projects')
def projects():
    return render_template('home_sections/newhome.html') #render_template('gen_ai.html')

@app.route('/prompts', defaults={'subsection': None, 'blog_slug': None})
@app.route('/prompts/<subsection>', defaults={'blog_slug': None})
@app.route('/prompts/<subsection>/<blog_slug>')
def prompts(subsection, blog_slug):
    base_path = 'prompts'
    print(subsection, blog_slug)

    # Load all prompt sections (e.g., ['Image Generation', 'Novel Writing', ...])
    prompt_sections = [
        name for name in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, name)) and name != '.DS_Store'
    ]

    section_blogs = []
    blog_details = None
    latest_blogs_per_section = {}

    if subsection:
        subsection_path = os.path.join(base_path, subsection)

        if not os.path.exists(subsection_path):
            abort(404)

        # Load all blogs in that subsection
        for blog_dir in os.listdir(subsection_path):
            if blog_dir == '.DS_Store':
                continue
            blog_folder = os.path.join(subsection_path, blog_dir)
            meta_path = os.path.join(blog_folder, 'blog_meta.json')
            if os.path.exists(meta_path):
                with open(meta_path, 'r') as f:
                    meta = json.load(f)[0]
                    meta['slug'] = subsection
                    section_blogs.append(meta)

        # If a specific blog is selected
        if blog_slug:
            blog_folder = os.path.join(subsection_path, blog_slug)
            content_path = os.path.join(blog_folder, 'blog_content.md')
            meta_path = os.path.join(blog_folder, 'blog_meta.json')

            if not os.path.exists(content_path) or not os.path.exists(meta_path):
                abort(404)

            with open(content_path, 'r', encoding='utf-8') as f:
                blog_content = f.read()
            with open(meta_path, 'r') as f:
                blog_meta = json.load(f)

            blog_details = {
                'blog_content': blog_content,
                'blog_meta': blog_meta,
                'blog_slug': blog_slug,
                'subsection': subsection
            }

    else:

        section_blogs = []  # clear or build fresh list per section

        for section in prompt_sections:
            subsection_path = os.path.join(base_path, section)
            blogs = []
            # Load all blogs in section
            for blog_dir in os.listdir(subsection_path):
                if blog_dir == '.DS_Store':
                    continue
                blog_folder = os.path.join(subsection_path, blog_dir)
                meta_path = os.path.join(blog_folder, 'blog_meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path, 'r') as f:
                        meta = json.load(f)[0]
                        meta['slug'] = blog_dir
                        blogs.append(meta)

            # Sort blogs by date (assuming 'date' key exists in meta)
            blogs_sorted = sorted(blogs, key=lambda x: x.get('date', ''), reverse=True)

            # Store top 3 latest blogs per section in the dictionary
            latest_blogs_per_section[section] = blogs_sorted[:3]


    return render_template(
        'prompts/prompts.html',
        all_section_names=prompt_sections,
        all_blogs_of_particular_section=section_blogs,
        latest_blogs_per_section=latest_blogs_per_section,
        blog_details=blog_details
    )


@app.route('/cheatsheets', defaults={'path': ''})
@app.route('/cheatsheets/', defaults={'path': ''})
@app.route('/cheatsheets/<path:path>')
def cheatsheets(path):
    def get_folder_structure(root_dir, rel_path=""):
        structure = []
        for item in sorted(os.listdir(root_dir)):
            if item == '.DS_Store':
                continue
            path_full = os.path.join(root_dir, item)
            rel_item_path = os.path.join(rel_path, item)
            if os.path.isdir(path_full):
                structure.append({
                    "type": "folder",
                    "name": item,
                    "path": rel_item_path,
                    "children": get_folder_structure(path_full, rel_item_path)
                })
            else:
                structure.append({
                    "type": "file",
                    "name": item,
                    "path": rel_item_path
                })
        return structure

    cheatsheets_dir = os.path.join(app.root_path, "cheatsheets")
    sidebar_structure = get_folder_structure(cheatsheets_dir)

    # Absolute safe path
    target_path = os.path.abspath(os.path.join(cheatsheets_dir, path))
    if not target_path.startswith(cheatsheets_dir):
        abort(403)  # Prevent directory traversal

    file_content = None
    if os.path.isfile(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

    return render_template(
        'cheatsheets/cheatsheets.html',
        sidebar=sidebar_structure,
        page_prefix = "cheatsheets",
        initial_path=path,
        file_content=file_content
    )


@app.route('/cheatsheets/<path:filename>')
def get_cheatsheet_file(filename):
    cheatsheets_dir = os.path.join(app.root_path, "cheatsheets")
    file_path = os.path.abspath(os.path.join(cheatsheets_dir, filename))

    if not file_path.startswith(cheatsheets_dir):
        abort(403)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    else:
        abort(404)

if __name__ == '__main__':
    app.run(debug=True)
