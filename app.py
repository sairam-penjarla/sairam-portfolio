from flask import Flask, render_template
from flask import Flask, render_template, abort
import os
import re
from markupsafe import Markup

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/resources')
def resources():
    return render_template('resources.html')

@app.route('/gen-ai')
def gen_ai():
    return render_template('gen_ai.html')

@app.route('/courses')
def courses():
    return render_template('courses.html')

# Add free courses route
@app.route('/free-courses')
def free_courses_home():
    # Get the root structure of the courses directory
    sidebar_structure = get_courses_structure()
    return render_template('free_courses.html', sidebar=sidebar_structure, content="# Free Courses\n\nSelect a course from the sidebar to get started.", breadcrumbs=[{"name": "Free Courses", "url": "/free-courses"}])


def extract_metadata(md_text):
    metadata = {}
    meta_match = re.search(r'<!--(.*?)-->', md_text, re.DOTALL)
    if meta_match:
        meta_lines = meta_match.group(1).strip().split('\n')
        for line in meta_lines:
            key, value = line.split(':', 1)
            metadata[key.strip()] = value.strip()
    return metadata

def get_blog_structure(active_category=None, active_slug=None):
    base_path = 'blogs'
    structure = []

    for category in sorted(os.listdir(base_path)):
        cat_path = os.path.join(base_path, category)
        if not os.path.isdir(cat_path):
            continue
        blogs = []
        for filename in sorted(os.listdir(cat_path)):
            if filename.endswith('.md'):
                with open(os.path.join(cat_path, filename), 'r') as f:
                    meta = extract_metadata(f.read())
                slug = filename.replace('.md', '')
                blogs.append({
                    'title': meta.get('title', slug),
                    'slug': slug,
                    'active': category == active_category and slug == active_slug
                })
        structure.append({
            'name': category,
            'expanded': category == active_category,
            'blogs': blogs
        })
    return structure

# Function to get course structure with nested folders
def get_courses_structure(path=None, active_path=None, active_slug=None):
    """
    Recursively get the structure of the courses directory with support for nested folders
    """
    base_path = 'courses' if path is None else path
    structure = []
    
    try:
        # Define the sorting function for module/part files
        def get_module_number(filename):
            # Extract module number for sorting
            import re
            
            # Extract numbers from various patterns:
            # 1. For Module X pattern
            match = re.search(r'Module\s+(\d+)', filename, re.IGNORECASE)
            if match:
                return int(match.group(1))
            
            # 2. For part X pattern
            part_match = re.search(r'part\s*(\d+)', filename, re.IGNORECASE)
            if part_match:
                return int(part_match.group(1))
            
            # 3. For tutorial X pattern (handling variable spaces)
            tutorial_match = re.search(r'tutorial\s*(\d+)', filename, re.IGNORECASE)
            if tutorial_match:
                return int(tutorial_match.group(1))
                
            # 4. Last resort: find any number in the filename
            number_match = re.search(r'(\d+)', filename)
            if number_match:
                return int(number_match.group(1))
                
            return 0  # Default value if pattern doesn't match

        # Get root-level .md files
        root_md_files = []
        folders_to_process = []
        
        # First pass: separate files and folders
        for item in sorted(os.listdir(base_path)):
            item_path = os.path.join(base_path, item)
            
            # Skip hidden files/folders
            if item.startswith('.'):
                continue
            
            if os.path.isdir(item_path):
                folders_to_process.append(item)
            elif item.endswith('.md'):
                root_md_files.append(item)
                
        # Process root-level .md files
        if root_md_files and base_path == 'courses':
            # Sort files numerically
            root_md_files.sort(key=get_module_number)
            
            # Create a special "Root" folder for these files
            root_folder = {
                'name': 'Course Materials',
                'expanded': active_path is None,
                'courses': [],
                'subfolders': []
            }
            
            # Add each root .md file
            for filename in root_md_files:
                md_path = os.path.join(base_path, filename)
                with open(md_path, 'r') as f:
                    meta = extract_metadata(f.read())
                
                slug = filename.replace('.md', '')
                root_folder['courses'].append({
                    'title': meta.get('title', slug),
                    'slug': slug,
                    'active': active_slug == slug and active_path is None
                })
            
            structure.append(root_folder)
                
        # Process folders alphabetically
        for item in folders_to_process:
            item_path = os.path.join(base_path, item)
            folder_name = os.path.basename(item_path)
            
            # Get current path relative to courses directory
            rel_path = os.path.relpath(item_path, 'courses')
            is_active = active_path and rel_path in active_path
            
            # Initialize folder structure
            folder = {
                'name': folder_name,
                'expanded': is_active,
                'courses': [],
                'subfolders': get_courses_structure(item_path, active_path, active_slug)
            }
            
            # Get all .md files
            md_files = []
            for filename in os.listdir(item_path):
                if filename.endswith('.md'):
                    md_files.append(filename)
            
            # Apply custom sorting for module files
            md_files.sort(key=get_module_number)
            
            # Add .md files in this directory as courses
            for filename in md_files:
                md_path = os.path.join(item_path, filename)
                with open(md_path, 'r') as f:
                    meta = extract_metadata(f.read())
                
                slug = filename.replace('.md', '')
                folder['courses'].append({
                    'title': meta.get('title', slug),
                    'slug': slug,
                    'active': active_slug == slug and rel_path in active_path
                })
            
            structure.append(folder)
    except FileNotFoundError:
        # Directory doesn't exist yet, return empty structure
        pass
        
    return structure

# Helper function to build breadcrumbs for nested course paths
def build_breadcrumbs(path):
    parts = path.strip('/').split('/')
    crumbs = []
    current_path = ''
    
    # Add home link
    crumbs.append({"name": "Free Courses", "url": "/free-courses"})
    
    # Add each path segment
    for part in parts:
        if part:  # Skip empty parts
            current_path += '/' + part
            crumbs.append({"name": part, "url": "/free-courses" + current_path})
    
    return crumbs


@app.route('/projects/')
def projects():
    return render_template('projects_home.html')


@app.route('/projects/<category>')
def projects_category(category=None):
    blog_dir = os.path.join('blogs', category)
    blog_entries = []

    for filename in os.listdir(blog_dir):
        if filename.endswith('.md'):
            # Use the filename itself without extracting metadata
            slug = filename.replace('.md', '')
            blog_entries.append({
                'slug': slug,
                'title': slug,  # Use slug as title instead of metadata
                'date': '',     # Empty date since we're not using metadata
                'thumbnail': '' # Empty thumbnail since we're not using metadata
            })

    return render_template('projects.html', category=category, blogs=blog_entries)

@app.route('/projects/<category>/<slug>')
def view_blog(category, slug):
    path = os.path.join('blogs', category, slug + '.md')
    if not os.path.exists(path):
        abort(404)

    with open(path, 'r') as f:
        content = f.read()
        metadata = extract_metadata(content)
        html_content = content.split('-->')[-1]
        
        # Mark content as safe to prevent HTML encoding
        html_content = Markup(html_content)

    sidebar_structure = get_blog_structure(category, slug)

    return render_template(
        'blog.html',
        content=html_content,
        metadata=metadata,
        category=category,
        sidebar=sidebar_structure
    )

# Route to handle nested course paths
@app.route('/free-courses/<path:course_path>')
def view_course_path(course_path):
    # Check if the path is a folder or a file
    full_path = os.path.join('courses', course_path)
    
    if os.path.isdir(full_path):
        # It's a folder, show the folder contents
        breadcrumbs = build_breadcrumbs(course_path)
        sidebar_structure = get_courses_structure(active_path=course_path)
        
        # Create a welcome message for the folder
        content = f"# {os.path.basename(course_path).replace('-', ' ').title()}\n\nSelect a course from the sidebar to get started."
        
        return render_template(
            'free_courses.html',
            content=content,
            sidebar=sidebar_structure,
            breadcrumbs=breadcrumbs
        )
    else:
        # It's a file (or should be), extract the folder and file parts
        folder_path = os.path.dirname(course_path)
        filename = os.path.basename(course_path)
        
        # Check if it's a markdown file without the .md extension
        md_path = os.path.join('courses', folder_path, filename + '.md')
        if os.path.exists(md_path):
            with open(md_path, 'r') as f:
                content = f.read()
                metadata = extract_metadata(content)
                html_content = content.split('-->')[-1] if '-->' in content else content
                
                # Mark content as safe to prevent HTML encoding
                html_content = Markup(html_content)
            
            sidebar_structure = get_courses_structure(active_path=folder_path, active_slug=filename)
            breadcrumbs = build_breadcrumbs(course_path)
            
            return render_template(
                'free_courses.html',
                content=html_content,
                metadata=metadata,
                sidebar=sidebar_structure,
                breadcrumbs=breadcrumbs
            )
        else:
            abort(404)


if __name__ == '__main__':
    app.run(debug=True)
