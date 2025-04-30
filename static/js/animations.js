// Intersection Observer for animations
document.addEventListener('DOMContentLoaded', function() {
    // Function to check if element is in viewport and apply animations
    const animateOnScroll = () => {
        // Target animate-section elements
        const sections = document.querySelectorAll('.animate-section');
        
        // Create Intersection Observer
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                // If section is in viewport
                if (entry.isIntersecting) {
                    // Add visible class to section
                    entry.target.classList.add('visible');
                    
                    // Get all staggered items within this section
                    const staggeredItems = entry.target.querySelectorAll('.staggered-item');
                    
                    // Add visible class to each item with increasing delay
                    staggeredItems.forEach((item, index) => {
                        setTimeout(() => {
                            item.classList.add('visible');
                        }, 150 * index); // 150ms stagger between each item
                    });
                }
            });
        }, {
            threshold: 0.1 // Trigger when 10% of element is visible
        });
        
        // Observe all sections
        sections.forEach(section => {
            observer.observe(section);
        });
    };
    
    // Specific animations for projects section
    const animateProjects = () => {
        const projectsSection = document.getElementById('projects-section');
        
        if (projectsSection) {
            const projectCards = projectsSection.querySelectorAll('.project-card');
            
            projectCards.forEach(card => {
                card.classList.add('staggered-item');
            });
        }
    };
    
    // Initialize animations
    animateOnScroll();
    animateProjects();
}); 