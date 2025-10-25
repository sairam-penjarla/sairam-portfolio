from flask import Flask, render_template
from flask import Flask, render_template, abort
import os
import re
from flask import Flask, render_template, jsonify, send_file, abort
import os
import random
import os
import importlib.util
import json
from markupsafe import Markup
import os
import importlib.util
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('about_me.html')

@app.route('/roadmap')
def roadmap():
    return render_template('roadmap/roadmap.html')

@app.route('/error')
def error():
    return render_template('404_error.html')

@app.route('/test')
def test():
    return render_template('test.html')

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() 
            for text in re.split(r'(\d+)', s)]


def load_course_data(py_file_path):
    """Dynamically load course_data from a .py file"""
    spec = importlib.util.spec_from_file_location("module.name", py_file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "course_data", None)

    
@app.route('/learn')
@app.route('/learn/<var0>')
@app.route('/learn/<var0>/<var1>')
@app.route('/learn/<var0>/<var1>/<var2>')
@app.route('/learn/<var0>/<var1>/<var2>/<var3>')
def learn(var0='courses', var1='All', var2=None, var3=None):
    if not var2:
        
        root = "courses"
        all_courses = []

        for category in os.listdir(root):
            category_path = os.path.join(root, category)
            if not os.path.isdir(category_path):
                continue

            for course in os.listdir(category_path):
                course_path = os.path.join(category_path, course)
                if not os.path.isdir(course_path):
                    continue

                # Find .py files in this course
                py_files = [f for f in os.listdir(course_path) if f.endswith(".py")]
                if not py_files:
                    continue

                # Pick just ONE (the first, alphabetically)
                py_file = sorted(py_files)[0]
                py_file_path = os.path.join(course_path, py_file)
                # print(py_file_path)
                course_data = load_course_data(py_file_path)
                if course_data:
                    all_courses.append(course_data)
        
        if var0 not in ['courses', 'roadmap']:
            var0 = "courses"
        filters = list(set(["_".join(x['category'].lower().split(" ")) for x in all_courses]))
        if var1 not in filters:
            var1 = "All"
        return render_template('learn.html', var0=var0, var1=var1, var2=var2, all_courses=all_courses)
    else:
        if var3:
            result = None
            py_file_path = os.path.join(var0, var1, var2)
            for fl in os.listdir(py_file_path):
                if fl.endswith(".py"):
                    py_fl_file_path = os.path.join(var0, var1, var2, fl)
                    course_data = load_course_data(py_fl_file_path)

                    if var3 == "_".join(course_data['module_title'].lower().split(" ")):
                        course_data['title'] = course_data['module_title']
                        result = course_data

            blog_data = {
                "article" : result,
                "relatedPosts": [{
                    "url":"",
                    "category":"",
                    "title":"",
                    "date":""
                }]*3
            }
            categories_dict = []
            category=""
            blog=""
            page_heading = "blogs"
            return render_template('blog_post.html', page_heading=page_heading, category=category, blog=blog, blog_data=blog_data, categories_dict=categories_dict)
                
        root = "courses"
        course_url = f"/learn/{var0}/{var1}/{var2}"

        all_courses = []

        for category in os.listdir(root):
            category_path = os.path.join(root, category)
            if not os.path.isdir(category_path):
                continue

            for course in os.listdir(category_path):
                course_path = os.path.join(category_path, course)
                if not os.path.isdir(course_path):
                    continue

                # Find .py files in this course
                py_files = [f for f in os.listdir(course_path) if f.endswith(".py")]
                if not py_files:
                    continue

                # Pick just ONE (the first, alphabetically)
                py_file = sorted(py_files, key=natural_sort_key)
                for pf in py_file:
                    py_file_path = os.path.join(course_path, pf)
                    # print(py_file_path)
                    course_data = load_course_data(py_file_path)
                    if course_data:
                        if course_data['url'] == course_url:
                            course_data['type'] = "Video"
                            course_data['module'] = "_".join(course_data['module_title'].lower().split(" "))
                            all_courses.append(course_data)
        def reform(element):
            element["name"] = element['module_title']
            element["resources"] = []
            return element


        course_url = f"/learn/{var0}/{var1}"

        var1 = all_courses[0]['course_title']

        course = {
            all_courses[0]['course_title'] : {
                "modules" : [
                    reform(x) for x in all_courses
                ] # add resoources to the .py files
                }
        }
        
        
        return render_template('courses.html', var0=var0, var1=var1, var2=var2, course=course, course_url=course_url)



@app.route('/roadmap_json')
def roadmap_json():
    try:
        with open("roadmap.json", "r", encoding="utf-8") as file:
            try:
                roadmap = json.load(file)
                print(roadmap)
                return jsonify(roadmap), 200
            except json.JSONDecodeError as e:
                raise ValueError(f"Error decoding JSON file: {e}")
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch projects',
            'message': str(e)
        }), 500
    

@app.route('/resources')
def resources():
    return render_template('resources.html')


@app.route('/cheatsheets')
@app.route('/cheatsheets/')
@app.route('/cheatsheets/<category>')
def cheatsheets(category=None):
        
    BASE_DIR = "cheatsheets"

    def load_py_course_data(py_path):
        """Dynamically load a .py file and return its course_data dictionary"""
        spec = importlib.util.spec_from_file_location("course_module", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.course_data

    def get_all_courses(base_dir=BASE_DIR):
        """Read all .py files in cheatsheets and return a list of dictionaries"""
        courses_list = []

        for root, dirs, files in os.walk(base_dir):
            py_files = [f for f in files if f.endswith(".py")]
            py_files = sorted(py_files, key=natural_sort_key)
            for py_file in py_files:
                py_path = os.path.join(root, py_file)
                try:
                    course_data = load_py_course_data(py_path)
                    course_dict = {
                        "category": course_data["category"],
                        "title": course_data["module_title"],
                        "date": course_data["date"],
                        "imageUrl": course_data["imageUrl"],
                        "url": course_data["url"].replace("/learn/courses/", "/cheatsheets/")
                    }
                    courses_list.append(course_dict)
                except Exception as e:
                    print(f"Error loading {py_path}: {e}")

        return courses_list

    articles_list = get_all_courses()
    page_heading = "Cheatsheets"
    # Render the same template for /blogs and /blogs/<category>
    return render_template('blog.html', page_heading=page_heading, articles_list=articles_list, category=category)

@app.route('/blogs')
@app.route('/blogs/')
@app.route('/blogs/<category>')
def blogs(category=None):
    base_dir = "blogs"
    articles_list = []

    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):
            if filename.endswith(".json"):
                json_path = os.path.join(category_path, filename)

                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                article = data.get("article", {})
                articles_list.append({
                    "category": article.get("category", category),
                    "title": article.get("title", ""),
                    "date": article.get("date", ""),
                    "imageUrl": article.get("imageUrl", ""),
                    "url": f"/blogs/{category}/{filename[:-5]}"
                })
    articles_list.sort(
        key=lambda x: datetime.strptime(x["date"], "%b %d, %Y") if x["date"] else datetime.min,
        reverse=True  # set to False for oldest first
    )
    # Render the same template for /blogs and /blogs/<category>
    page_heading="Blogs"
    return render_template('blog.html', page_heading=page_heading, articles_list=articles_list, category=category)

@app.route('/blogs/<category>/<blog>')
def blog_post(category, blog):
    # Render blog post page
    def get_blog_data(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # --- Load main blog JSON ---
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        article = data.get("article", {})

        # --- Identify folder and other JSON files ---
        folder = os.path.dirname(file_path)
        all_files = [
            f for f in os.listdir(folder)
            if f.endswith(".json") and f != os.path.basename(file_path)
        ]

        related_posts = []

        # --- Pick up to 3 random related posts ---
        if all_files:
            sample_files = random.sample(all_files, min(3, len(all_files)))
            for rel_file in sample_files:
                rel_path = os.path.join(folder, rel_file)
                with open(rel_path, "r", encoding="utf-8") as f:
                    rel_data = json.load(f)
                rel_article = rel_data.get("article", {})
                related_posts.append({
                    "title": rel_article.get("title", ""),
                    "category": rel_article.get("category", ""),
                    "date": rel_article.get("date", ""),
                    "url": f"{rel_path[:-5]}"
                })
                
        # --- Final structure ---
        result = {
            "article": {
                "title": article.get("title", ""),
                "category": article.get("category", ""),
                "date": article.get("date", ""),
                "readTime": article.get("readTime", ""),
                "imageUrl": article.get("imageUrl", ""),
                "body": article.get("body", ""),
            },
            "relatedPosts": related_posts
        }

        return result


    # Example usage:
    file_path = f"blogs/{category}/{blog}.json"
    blog_data = get_blog_data(file_path)

    md_file = blog_data["article"]["body"]
    with open(f"blogs/{category}/{md_file}", "r", encoding="utf-8") as f:
        blog_data["article"]["body"] = f.read()


    base_dir = "blogs"
    categories_dict = {"categories": []}

    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        links = []

        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_path = os.path.join(folder_path, filename)
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                title = data.get("article", {}).get("title", "")
                file_url = f"/blogs/{folder_name}/{filename.replace('.json', '')}"

                links.append({
                    "text": title,
                    "url": file_url
                })

        categories_dict["categories"].append({
            "name": folder_name,
            "links": links
        })
    page_heading = "blogs"
    return render_template('blog_post.html', page_heading=page_heading, category=category, blog=blog, blog_data=blog_data, categories_dict=categories_dict)


@app.route('/cheatsheets/<category>/<blog>')
def cheatsheets_post(category, blog):
    import os
    import importlib.util
    import random

    BASE_DIR = "cheatsheets"

    def load_py_course_data(py_path):
        """Dynamically load a .py file and return its course_data dictionary"""
        spec = importlib.util.spec_from_file_location("course_module", py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.course_data

    def build_categories_dict(base_dir=BASE_DIR):
        """
        Returns a dictionary of all categories and their links
        {
            "categories": [
                {
                    "name": "Folder Name",
                    "links": [
                        {"text": "Module Title", "url": "/cheatsheets/..."},
                        ...
                    ]
                },
                ...
            ]
        }
        """
        categories = []

        for folder in os.listdir(base_dir):
            folder_path = os.path.join(base_dir, folder)
            if os.path.isdir(folder_path):
                links = []
                py_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
                py_files = sorted(py_files, key=natural_sort_key)
                for py_file in py_files:
                    py_path = os.path.join(folder_path, py_file)
                    try:
                        course_data = load_py_course_data(py_path)
                        links.append({
                            "text": course_data.get("module_title", ""),
                            "url": course_data.get("url", "").replace("/learn/courses/", "/cheatsheets/")
                        })
                    except Exception as e:
                        print(f"Error loading {py_path}: {e}")

                categories.append({
                    "name": folder.replace("_", " ").title(),
                    "links": links
                })

        return {"categories": categories}


    def get_article_with_related(category, blog, base_dir=BASE_DIR):
        """
        Returns both:
        - res: a dictionary containing the article and related posts
        - categories_dict: dictionary containing all categories and their links
        """
        categories_dict = build_categories_dict(base_dir)

        folder_path = os.path.join(base_dir, category)
        if not os.path.exists(folder_path):
            print(f"Category folder '{category}' does not exist.")
            return None, categories_dict

        constructed_url = f"/learn/courses/{category}/{blog}"

        article_data = None
        all_courses = []

        # Load all .py files in the folder
        py_files = [f for f in os.listdir(folder_path) if f.endswith(".py")]
        for py_file in py_files:
            py_path = os.path.join(folder_path, py_file)
            try:
                course_data = load_py_course_data(py_path)
                all_courses.append(course_data)
                if course_data.get("url") == constructed_url:
                    article_data = course_data
            except Exception as e:
                print(f"Error loading {py_path}: {e}")

        if not article_data:
            print(f"No article found for URL: {constructed_url}")
            return None, categories_dict

        # Prepare related posts (randomly pick 3 different from main article)
        other_courses = [c for c in all_courses if c.get("url") != constructed_url]
        related_posts = []
        sample_count = min(3, len(other_courses))
        if sample_count > 0:
            for c in random.sample(other_courses, sample_count):
                related_posts.append({
                    "title": c.get("module_title", ""),
                    "category": c.get("category", ""),
                    "date": c.get("date", ""),
                    "url": c.get("url", "").replace("/learn/courses/", "cheatsheets/")
                })

        # Build result dictionary
        res = {
            "article": {
                "title": article_data.get("module_title", ""),
                "category": article_data.get("category", ""),
                "date": article_data.get("date", ""),
                "readTime": article_data.get("readTime", ""),  # optional field
                "imageUrl": article_data.get("imageUrl", ""),
                "body": article_data.get("body", "")
            },
            "relatedPosts": related_posts
        }

        return res, categories_dict

    blog_data, categories_dict = get_article_with_related(category, blog)
    page_heading = "cheatsheet"

    return render_template('blog_post.html', page_heading=page_heading, category=category, blog=blog, blog_data=blog_data, categories_dict=categories_dict)


# @app.route('/blogs', defaults={'path': ''})
# @app.route('/blogs/', defaults={'path': ''})
# @app.route('/blogs/<path:path>')
# def blogs(path):
#     def get_folder_structure(root_dir, rel_path=""):
#         structure = []
#         for item in sorted(os.listdir(root_dir)):
#             if item == '.DS_Store':
#                 continue
#             path_full = os.path.join(root_dir, item)
#             rel_item_path = os.path.join(rel_path, item)
#             if os.path.isdir(path_full):
#                 structure.append({
#                     "type": "folder",
#                     "name": item,
#                     "path": rel_item_path,
#                     "children": get_folder_structure(path_full, rel_item_path)
#                 })
#             else:
#                 structure.append({
#                     "type": "file",
#                     "name": item,
#                     "path": rel_item_path
#                 })
#         return structure

#     blogs_dir = os.path.join(app.root_path, "blogs")
#     sidebar_structure = get_folder_structure(blogs_dir)

#     # Absolute safe path
#     target_path = os.path.abspath(os.path.join(blogs_dir, path))
#     if not target_path.startswith(blogs_dir):
#         abort(403)  # Prevent directory traversal

#     file_content = None
#     if os.path.isfile(target_path):
#         with open(target_path, 'r', encoding='utf-8') as f:
#             file_content = f.read()

#     return render_template(
#         'blogs/blogs.html',
#         page_prefix = "blogs",
#         sidebar=sidebar_structure,
#         initial_path=path,
#         file_content=file_content
#     )


# @app.route('/blog/<path:filename>')
# def get_blog_file(filename):
#     blogs_dir = os.path.join(app.root_path, "blogs")
#     file_path = os.path.abspath(os.path.join(blogs_dir, filename))

#     if not file_path.startswith(blogs_dir):
#         abort(403)

#     if os.path.isfile(file_path):
#         with open(file_path, 'r', encoding='utf-8') as f:
#             content = f.read()
#         return jsonify({"content": content})
#     else:
#         abort(404)

        
@app.route('/projects')
def projects():
    with open("projects/projects.json", "r", encoding="utf-8") as file:
        projects = json.load(file)
    
    markdown_projects = {}

    with open("projects/projects.json", "r", encoding="utf-8") as file:
        projects = json.load(file)

    for p in projects:
        key = p['md_file']
        with open(key, 'r', encoding='utf-8') as file:
            markdown_projects[key] = file.read()

    return render_template('projects.html', projects=projects, markdown_projects=markdown_projects)



@app.route('/projects_json', methods=['GET'])
def get_projects():
    """
    API endpoint to fetch all projects
    Returns JSON array of project objects
    """
    try:
        # Here you would typically fetch from your database
        # For now, returning sample data
        projects = get_projects_from_database()  # Replace with your data source
        
        return jsonify(projects), 200
    except Exception as e:
        return jsonify({
            'error': 'Failed to fetch projects',
            'message': str(e)
        }), 500

def get_projects_from_database(file_path="projects/projects.json"):
    """
    Reads projects data from a JSON file and enriches each project with details from corresponding .md files.
    The JSON should contain a 'md_file' key with the path to the markdown file for each project.
    
    :param file_path: Path to the JSON file containing project data.
    :return: List of project dictionaries with 'details' key containing markdown content.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Projects JSON file not found at: {file_path}")

    with open(file_path, "r", encoding="utf-8") as file:
        try:
            projects = json.load(file)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error decoding JSON file: {e}")

    # Read markdown files for each project
    for project in projects:
        md_file_path = project.get('md_file')
        
        if md_file_path and os.path.exists(md_file_path):
            try:
                with open(md_file_path, "r", encoding="utf-8") as md_file:
                    project['details'] = md_file.read()
            except Exception as e:
                print(f"Warning: Error reading markdown file {md_file_path}: {e}")
                project['details'] = ""
        else:
            project['details'] = ""
            if md_file_path:
                print(f"Warning: Markdown file not found: {md_file_path}")

    return projects

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




@app.route('/courses', defaults={'path': ''})
@app.route('/courses/', defaults={'path': ''})
@app.route('/courses/<path:path>')
def courses(path):
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

    courses_dir = os.path.join(app.root_path, "courses")
    sidebar_structure = get_folder_structure(courses_dir)

    # Absolute safe path
    target_path = os.path.abspath(os.path.join(courses_dir, path))
    if not target_path.startswith(courses_dir):
        abort(403)  # Prevent directory traversal

    file_content = None
    if os.path.isfile(target_path):
        with open(target_path, 'r', encoding='utf-8') as f:
            file_content = f.read()

    return render_template(
        'courses/courses.html',
        sidebar=sidebar_structure,
        page_prefix = "courses",
        initial_path=path,
        file_content=file_content
    )


@app.route('/courses/<path:filename>')
def get_courses_file(filename):
    courses_dir = os.path.join(app.root_path, "courses")
    file_path = os.path.abspath(os.path.join(courses_dir, filename))

    if not file_path.startswith(courses_dir):
        abort(403)

    if os.path.isfile(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return jsonify({"content": content})
    else:
        abort(404)
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
