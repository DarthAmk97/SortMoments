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
 * Fetch download count from GitHub Releases API
 */
async function fetchDownloadCount() {
    const countElement = document.getElementById('downloadCount');

    // Check cache first
    const cached = getCachedCount();
    if (cached !== null) {
        updateCountDisplay(cached);
        return;
    }

    try {
        const response = await fetch(
            `https://api.github.com/repos/${CONFIG.githubOwner}/${CONFIG.githubRepo}/releases`,
            {
                headers: {
                    'Accept': 'application/vnd.github.v3+json'
                }
            }
        );

        if (!response.ok) {
            throw new Error(`GitHub API returned ${response.status}`);
        }

        const releases = await response.json();

        // Sum up all asset download counts across all releases
        let totalDownloads = 0;
        releases.forEach(release => {
            if (release.assets) {
                release.assets.forEach(asset => {
                    totalDownloads += asset.download_count || 0;
                });
            }
        });

        // Cache the result
        setCachedCount(totalDownloads);

        // Update display
        updateCountDisplay(totalDownloads);

    } catch (error) {
        console.error('Error fetching download count:', error);
        // Show fallback text
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
 * Get cached download count
 */
function getCachedCount() {
    try {
        const cached = localStorage.getItem(CONFIG.cacheKey);
        if (!cached) return null;

        const { count, timestamp } = JSON.parse(cached);
        const now = Date.now();

        // Check if cache is still valid
        if (now - timestamp < CONFIG.cacheDuration) {
            return count;
        }

        // Cache expired
        localStorage.removeItem(CONFIG.cacheKey);
        return null;
    } catch (e) {
        return null;
    }
}

/**
 * Cache the download count
 */
function setCachedCount(count) {
    try {
        localStorage.setItem(CONFIG.cacheKey, JSON.stringify({
            count,
            timestamp: Date.now()
        }));
    } catch (e) {
        // localStorage might not be available
        console.warn('Could not cache download count:', e);
    }
}

/**
 * Track download button click (optional analytics)
 */
function trackDownload() {
    // You can add analytics tracking here if needed
    // For example: Google Analytics, Plausible, etc.
    console.log('Download initiated');
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
