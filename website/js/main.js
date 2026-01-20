/**
 * Sort Moments - Landing Page JavaScript
 */

// Configuration
const CONFIG = {
    // Update these with your actual GitHub username and repo name
    githubOwner: 'DarthAmk97',
    githubRepo: 'SortMoments',
    cacheKey: 'sortmoments_download_count',
    cacheDuration: 5 * 60 * 1000, // 5 minutes in milliseconds
};

/**
 * Fetch download count from server API
 */
async function fetchDownloadCount() {
    const countElement = document.getElementById('downloadCount');

    try {
        const response = await fetch('/api/counter');

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        const count = data.count || 0;

        // Update display
        updateCountDisplay(count);

    } catch (error) {
        console.error('Error fetching download count:', error);
        // Show fallback
        countElement.textContent = '0';
    }
}

/**
 * Update the download count display with animation
 */
function updateCountDisplay(count) {
    const countElement = document.getElementById('downloadCount');

    // Format number with commas
    const formattedCount = count.toLocaleString();

    // Animate the count
    animateValue(countElement, 0, count, 1000);
}

/**
 * Animate a number from start to end
 */
function animateValue(element, start, end, duration) {
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Ease out cubic
        const easeOut = 1 - Math.pow(1 - progress, 3);

        const current = Math.floor(start + (end - start) * easeOut);
        element.textContent = current.toLocaleString();

        if (progress < 1) {
            requestAnimationFrame(update);
        }
    }

    requestAnimationFrame(update);
}


/**
 * Track download button click and increment counter on server
 */
async function trackDownload() {
    try {
        const response = await fetch('/api/counter/increment');

        if (!response.ok) {
            throw new Error(`Server returned ${response.status}`);
        }

        const data = await response.json();
        const newCount = data.count || 0;

        // Update display with animation
        updateCountDisplay(newCount);

        console.log('Download initiated. Counter updated to:', newCount);
    } catch (error) {
        console.error('Error incrementing counter:', error);
    }
}

/**
 * Initialize the page
 */
function init() {
    // Fetch and display download count
    fetchDownloadCount();

    // Add click tracking to download button
    const downloadBtn = document.getElementById('downloadBtn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', trackDownload);
    }

    // Smooth scroll for anchor links (if any)
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Run when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}
