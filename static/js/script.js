document.addEventListener("DOMContentLoaded", function () {
    const hamburger = document.querySelector(".hamburger");
    const tabsContainer = document.querySelector(".tabs-container");
    const tabs = document.querySelectorAll(".tab");
    const dropdownMenu = document.querySelector(".dropdown-menu");
    const projectsTab = document.querySelector(".tab:nth-child(3)"); // Projects tab (3rd tab)

    // Function to position the dropdown under the Projects tab
    function positionDropdown() {
        if (window.innerWidth > 768) {
            const projectsRect = projectsTab.getBoundingClientRect();
            dropdownMenu.style.left = `${projectsRect.left}px`;
            dropdownMenu.style.transform = 'none';
        } else {
            dropdownMenu.style.left = '';
            dropdownMenu.style.transform = '';
        }
    }

    // Call on page load
    positionDropdown();

    // Call on window resize
    window.addEventListener('resize', positionDropdown);

    // Toggle mobile menu
    hamburger.addEventListener("click", function () {
        hamburger.classList.toggle("active");
        tabsContainer.classList.toggle("active");
    });

    // Handle tab clicks
    tabs.forEach((tab) => {
        tab.addEventListener("click", function (e) {
            // Don't prevent default navigation

            // Remove active class from all tabs
            tabs.forEach((t) => t.classList.remove("active"));

            // Add active class to clicked tab
            this.classList.add("active");

            // Show dropdown menu if Projects tab is clicked
            if (this === projectsTab) {
                // dropdownMenu.classList.add("show");
                // // If it's the projects dropdown, consider preventing default only here
                // if (window.innerWidth > 768) {
                //     e.preventDefault(); // Only prevent for project tab on desktop
                // }
            } else {
                dropdownMenu.classList.remove("show");
            }

            // Close mobile menu
            if (window.innerWidth <= 768) {
                hamburger.classList.remove("active");
                tabsContainer.classList.remove("active");
            }
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener("click", function (event) {
        if (!projectsTab.contains(event.target) && !dropdownMenu.contains(event.target)) {
            dropdownMenu.classList.remove("show");
        }
    });

    // Toggle dropdown on Projects tab hover for desktop
    if (window.innerWidth > 768) {
        projectsTab.addEventListener("mouseenter", function () {
            dropdownMenu.classList.add("show");
            positionDropdown(); // Reposition when showing
        });

        dropdownMenu.addEventListener("mouseleave", function () {
            dropdownMenu.classList.remove("show");
        });

        projectsTab.addEventListener("mouseleave", function (e) {
            if (!dropdownMenu.contains(e.relatedTarget)) {
                setTimeout(() => {
                    if (!dropdownMenu.matches(":hover")) {
                        dropdownMenu.classList.remove("show");
                    }
                }, 100);
            }
        });
    }
});
