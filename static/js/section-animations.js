// Animation Controller for all sections
document.addEventListener("DOMContentLoaded", function() {
    // Set up the Intersection Observer
    const observerOptions = {
        root: null,
        rootMargin: "0px",
        threshold: 0.15
    };

    // Create observer for general section animations
    const sectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const section = entry.target;
                
                // Add visible class to the main section
                section.classList.add('visible');
                
                // Find and animate child elements with delays
                const animatedChildren = section.querySelectorAll('.animate-left, .animate-right, .animate-bottom, .animate-scale');
                animatedChildren.forEach((child, index) => {
                    setTimeout(() => {
                        child.classList.add('visible');
                    }, index * 100); // 100ms staggered delay
                });
                
                // Find and animate items with specified delays
                const delayedItems = section.querySelectorAll('[class*="delay-"]');
                delayedItems.forEach(item => {
                    // Extract delay time from class (delay-100, delay-200, etc.)
                    const delayClass = Array.from(item.classList).find(cls => cls.startsWith('delay-'));
                    if (delayClass) {
                        const delay = parseInt(delayClass.split('-')[1]);
                        setTimeout(() => {
                            item.classList.add('visible');
                        }, delay);
                    }
                });
                
                // Optional: Stop observing after animation
                if (section.dataset.observeOnce !== "false") {
                    sectionObserver.unobserve(section);
                }
            }
        });
    }, observerOptions);

    // Create observer for staggered items
    const staggeredObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const container = entry.target;
                const items = container.querySelectorAll('.staggered-item');
                
                items.forEach((item, index) => {
                    setTimeout(() => {
                        item.classList.add('visible');
                    }, 50 * index); // 50ms staggered delay
                });
                
                // Optional: Stop observing after animation
                staggeredObserver.unobserve(container);
            }
        });
    }, observerOptions);

    // Observe all animated sections
    document.querySelectorAll('.animate-section').forEach(section => {
        sectionObserver.observe(section);
    });

    // Observe all staggered containers
    document.querySelectorAll('.staggered-container').forEach(container => {
        staggeredObserver.observe(container);
    });
}); 