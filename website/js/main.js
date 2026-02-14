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
 * Initialize mobile menu toggle
 */
function initMobileMenu() {
    const menuToggle = document.getElementById('mobileMenuToggle');
    const navLinks = document.getElementById('navLinks');

    if (!menuToggle || !navLinks) return;

    // Toggle menu on button click
    menuToggle.addEventListener('click', () => {
        menuToggle.classList.toggle('active');
        navLinks.classList.toggle('mobile-open');
    });

    // Close menu when a link is clicked
    navLinks.querySelectorAll('a').forEach(link => {
        link.addEventListener('click', () => {
            menuToggle.classList.remove('active');
            navLinks.classList.remove('mobile-open');
        });
    });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('nav')) {
            menuToggle.classList.remove('active');
            navLinks.classList.remove('mobile-open');
        }
    });
}

/**
 * ReactBits-inspired "shuffle" text animation for the hero title.
 * Runs once on load, and again on hover/focus for delight.
 */
function initShuffleTitle() {
    const el = document.querySelector('.shuffle-title[data-shuffle]');
    if (!el) return;

    const original = el.getAttribute('data-shuffle') || el.textContent || '';
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
    let timer = null;
    let loopTimer = null;

    // Tuning
    const TICK_MS = 42;
    const TOTAL_TICKS = 18; // total shuffle duration ~= TICK_MS * TOTAL_TICKS (~750ms)
    const PAUSE_AFTER_DONE_MS = 5000;

    function pickIndicesForCycle(text) {
        // Pick 2 indices per "word" (runs of letters/numbers). Keep punctuation/spaces stable.
        const picks = new Set();
        const reWord = /[A-Za-z0-9]+/g;
        let m;
        while ((m = reWord.exec(text)) !== null) {
            const start = m.index;
            const end = start + m[0].length;
            const len = end - start;
            const count = Math.min(2, len);
            const chosen = new Set();
            while (chosen.size < count) {
                chosen.add(start + Math.floor(Math.random() * len));
            }
            for (const idx of chosen) picks.add(idx);
        }
        return picks;
    }

    function runShuffle() {
        // Respect reduced motion
        if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
            el.textContent = original;
            return;
        }

        if (timer) window.clearInterval(timer);

        const target = original;
        const hot = pickIndicesForCycle(target);
        let tick = 0;

        timer = window.setInterval(() => {
            const next = target
                .split('')
                .map((ch, idx) => {
                    if (!hot.has(idx)) return ch; // only fuzz selected indices
                    // Settle back to the correct characters towards the end of the run
                    if (tick >= TOTAL_TICKS - 4) return ch;
                    return chars[Math.floor(Math.random() * chars.length)];
                })
                .join('');

            el.textContent = next;
            tick += 1;

            if (tick >= TOTAL_TICKS) {
                window.clearInterval(timer);
                timer = null;
                el.textContent = target;
            }
        }, TICK_MS);
    }

    // Run shortly after load to avoid fighting initial fade-in
    window.setTimeout(() => {
        runShuffle();
        // Loop: after each run finishes, pause 5s, then rerun (with new random picks)
        const estimatedRunMs = TOTAL_TICKS * TICK_MS;
        const scheduleNext = () => {
            loopTimer = window.setTimeout(() => {
                runShuffle();
                scheduleNext();
            }, estimatedRunMs + PAUSE_AFTER_DONE_MS);
        };
        if (!loopTimer) scheduleNext();
    }, 260);

    // Replay on hover/focus (keyboard accessible)
    el.addEventListener('mouseenter', runShuffle);
    el.addEventListener('focus', runShuffle);
}

/**
 * Initialize the page
 */
function init() {
    // Initialize mobile menu
    initMobileMenu();

    // Hero title animation
    initShuffleTitle();

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
