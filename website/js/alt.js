(function () {
    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const button = document.querySelector('.menu-button');
    const nav = document.getElementById('siteNav');
    const languageToggle = document.querySelector('[data-language-toggle]');
    let currentLanguage = getInitialLanguage();
    let refreshCarouselLanguage = null;
    let refreshMobileStacks = null;
    let latestStats = null;

    const copy = {
        en: {
            skip: 'Skip to content',
            primaryNav: 'Primary navigation',
            menu: 'Menu',
            navWorkflow: 'Workflow',
            navPrivacy: 'Privacy',
            navDevelopers: 'Developers',
            navDemo: 'Demo',
            navCli: 'SortMoments CLI',
            languageToggle: 'اردو',
            languageButtonLabel: 'Switch to Urdu',
            heroOverline: 'Local photo sorting',
            heroTitle: 'Organize photos by person.',
            heroText: 'Point it at a folder. It groups matching faces and gives you normal folders back. No signup wall. No cloud middleman pretending your memories need a subscription.',
            downloads: 'downloads',
            visitors: 'visitors',
            downloadWindows: 'Download for Windows',
            windowsSub: 'Windows 10 and 11',
            downloadMacos: 'Download for macOS',
            macosSub: 'Apple Silicon beta',
            factsLabel: 'Product facts',
            factsDownloadsNote: 'crazy right?',
            factsVisitorsNote: 'First-party site analytics. no accounts.',
            localDefault: 'Local by default',
            localDefaultNote: 'Your photos stay on your machine.',
            noAccount: 'No account',
            noAccountNote: 'Download it. Run it. Skip the portal dance.',
            workflowOverline: 'Workflow',
            workflowTitle: 'From camera-roll soup to named people.',
            workflowText: 'Three boring steps. That is the point. The app does the face grouping; you keep control of the names and files.',
            workflowOneTitle: 'Pick the mess.',
            workflowOneText: 'Old exports, trips, weddings, external drives. The usual evidence locker.',
            workflowTwoTitle: 'Rename the people.',
            workflowTwoText: 'The model makes groups. You decide what the folders are called.',
            workflowThreeTitle: 'Open the output folder.',
            workflowThreeText: 'Normal files. Normal folders. A radical position, apparently.',
            workflowQuipOne: 'do we even need this section? boring.',
            workflowQuipTwo: 'maybe this is the developer trying hard for SEO.',
            privacyOverline: 'Privacy',
            privacyTitle: 'Your photo library stays on your computer.',
            privacyTextOne: 'Sort Moments is built for the folders people avoid until a hard drive starts making noises: family events, old phones, trips, shared drives.',
            privacyTextTwo: 'It is not a magic vault. It is a local desktop app. That is the privacy story.',
            privacyQuipOne: 'perfect, I do not have to donate my camera roll to another vibe-coded app.',
            privacyQuipTwo: 'cloud sync? adorable. hard pass.',
            privacyQuipThree: 'the subscription economy has been informed your cousin\'s wedding photos are not a growth channel.',
            privacyQuipFour: 'finally, a privacy story that does not need three trackers and a blood sample.',
            privacyQuipFive: 'local processing. wild concept. almost like computers can compute.',
            privacyQuipSix: 'is this what vibe coding by an actual developer looks like? ts tuff 🥀',
            developersOverline: 'Developers',
            developersTitle: 'Use the sorter without the window.',
            developersQuipOne: 'wait what, this even has a library?',
            developersQuipTwo: 'no need for a GUI if you have a thing for CLIs. probably a Linux user.',
            developersQuipThree: 'import it, automate it, pretend this was always the plan.',
            pythonLibrary: 'Python Library',
            pythonLibraryText: 'Import the face-grouping pipeline in your own workflow.',
            pythonLibrarySoon: 'coming soon',
            commandLine: 'Command line',
            commandLineText: 'Run repeat jobs from a source checkout.',
            demoOverline: 'Demo',
            demoTitle: 'Watch the full flow once.',
            demoQuipOne: 'great, another demo video :sigh:',
            demoQuipTwo: 'watching it once is faster than pretending screenshots explain motion.',
            answersOverline: 'Answers',
            faqTitle: 'The short version.',
            faqOneTitle: 'Does Sort Moments upload photos?',
            faqOneText: 'No. Sorting happens locally on your desktop. The cloud does not need a copy of your family album to prove it has machine learning.',
            faqTwoTitle: 'What platforms are supported?',
            faqTwoText: 'Windows 10/11 and macOS Apple Silicon beta. The CLI and Python library are there if you would rather automate the job.',
            faqThreeTitle: 'Is it free?',
            faqThreeText: 'Yes. Sort Moments is free and open source under the MIT license. Suspiciously reasonable, but true.',
            footerText: 'Sort Moments is free, open source, and built by Abdullah Khawaja.',
            currentHomepage: 'Current homepage',
            privacyPolicy: 'Privacy Policy',
            archive: 'Archive',
            releases: 'Releases',
            carouselLabel: 'Sort Moments workflow screenshots',
            carouselControlsLabel: 'Workflow carousel controls',
            slideProcessingLabel: 'Show processing stage',
            slideRenameLabel: 'Show rename stage',
            slideOutputLabel: 'Show output stage',
            slideReviewLabel: 'Show review stage',
        },
        ur: {
            skip: 'مواد پر جائیں',
            primaryNav: 'مرکزی نیویگیشن',
            menu: 'مینو',
            navWorkflow: 'ورک فلو',
            navPrivacy: 'پرائیویسی',
            navDevelopers: 'ڈویلپرز',
            navDemo: 'ڈیمو',
            languageToggle: 'English',
            languageButtonLabel: 'Switch to English',
            heroOverline: 'مقامی فوٹو چھانٹی',
            heroTitle: 'تصاویر لوگوں کے حساب سے ترتیب دیں۔',
            heroText: 'ایک فولڈر منتخب کریں۔ یہ ملتے جلتے چہروں کو گروپ کرتا ہے اور آپ کو عام فولڈرز واپس دیتا ہے۔ سائن اپ کی دیوار نہیں۔ نہ کلاؤڈ کا درمیانی آدمی جو آپ کی یادوں کو سبسکرپشن بنا دے۔',
            downloads: 'ڈاؤن لوڈز',
            visitors: 'وزٹرز',
            downloadWindows: 'Windows کے لیے ڈاؤن لوڈ کریں',
            windowsSub: 'Windows 10 اور 11',
            downloadMacos: 'macOS کے لیے ڈاؤن لوڈ کریں',
            macosSub: 'Apple Silicon بیٹا',
            factsLabel: 'پروڈکٹ فیکٹس',
            factsDownloadsNote: 'crazy right?',
            factsVisitorsNote: 'first-party site analytics۔ accounts کے بغیر۔',
            localDefault: 'ڈیفالٹ طور پر مقامی',
            localDefaultNote: 'آپ کی تصاویر آپ کے کمپیوٹر پر رہتی ہیں۔',
            noAccount: 'اکاؤنٹ نہیں',
            noAccountNote: 'ڈاؤن لوڈ کریں۔ چلائیں۔ پورٹل ڈرامہ چھوڑیں۔',
            workflowOverline: 'ورک فلو',
            workflowTitle: 'کیمرا رول کے شور سے نام والے لوگ۔',
            workflowText: 'تین سادہ قدم۔ یہی بات ہے۔ ایپ چہرے گروپ کرتی ہے؛ نام اور فائلیں آپ کے قابو میں رہتی ہیں۔',
            workflowOneTitle: 'گڑبڑ منتخب کریں۔',
            workflowOneText: 'پرانی exports، سفر، شادیاں، external drives۔ معمول کا ثبوتی لاکر۔',
            workflowTwoTitle: 'لوگوں کے نام بدلیں۔',
            workflowTwoText: 'ماڈل گروپس بناتا ہے۔ فولڈرز کے نام آپ طے کرتے ہیں۔',
            workflowThreeTitle: 'آؤٹ پٹ فولڈر کھولیں۔',
            workflowThreeText: 'عام فائلیں۔ عام فولڈرز۔ بظاہر کافی انقلابی۔',
            workflowQuipOne: 'کیا یہ سیکشن واقعی چاہیے تھا؟ بورنگ۔',
            workflowQuipTwo: 'شاید developer SEO کے لیے زیادہ کوشش کر رہا ہے۔',
            privacyOverline: 'پرائیویسی',
            privacyTitle: 'آپ کی فوٹو لائبریری آپ کے کمپیوٹر پر رہتی ہے۔',
            privacyTextOne: 'Sort Moments ان فولڈرز کے لیے ہے جنہیں لوگ تب تک ٹالتے ہیں جب تک hard drive آوازیں نہ نکالنے لگے: family events، پرانے phones، trips، shared drives۔',
            privacyTextTwo: 'یہ کوئی جادوئی vault نہیں۔ یہ local desktop app ہے۔ پرائیویسی کی کہانی بس یہی ہے۔',
            privacyQuipOne: 'زبردست، اپنی camera roll کسی اور vibe-coded app کو donate نہیں کرنی پڑے گی۔',
            privacyQuipTwo: 'cloud sync؟ cute. مگر نہیں۔',
            privacyQuipThree: 'subscription economy کو اطلاع دے دی گئی ہے کہ cousin کی wedding photos growth channel نہیں ہیں۔',
            privacyQuipFour: 'آخر کار privacy story جسے تین trackers اور blood sample نہیں چاہیے۔',
            privacyQuipFive: 'local processing۔ کمال تصور۔ جیسے computers compute بھی کر سکتے ہیں۔',
            privacyQuipSix: 'کیا developer والی vibe coding ایسی ہوتی ہے؟ ts tuff 🥀',
            developersOverline: 'ڈویلپرز',
            developersTitle: 'sorter کو window کے بغیر استعمال کریں۔',
            developersQuipOne: 'رکیں، اس کی library بھی ہے؟',
            developersQuipTwo: 'GUI کی ضرورت نہیں اگر آپ کو CLI پسند ہے۔ شاید Linux user۔',
            developersQuipThree: 'import کریں، automate کریں، اور pretend کریں یہی plan تھا۔',
            pythonLibrary: 'Python لائبریری',
            pythonLibraryText: 'face-grouping pipeline کو اپنے workflow میں import کریں۔',
            commandLine: 'کمانڈ لائن',
            commandLineText: 'source checkout سے repeat jobs چلائیں۔',
            demoOverline: 'ڈیمو',
            demoTitle: 'پورا flow ایک بار دیکھیں۔',
            demoQuipOne: 'زبردست، ایک اور demo video :sigh:',
            demoQuipTwo: 'ایک بار دیکھنا screenshots سے motion سمجھانے کا drama کرنے سے تیز ہے۔',
            answersOverline: 'جوابات',
            faqTitle: 'مختصر بات۔',
            faqOneTitle: 'کیا Sort Moments تصاویر upload کرتا ہے؟',
            faqOneText: 'نہیں۔ sorting آپ کے desktop پر مقامی طور پر ہوتی ہے۔ cloud کو آپ کے family album کی copy صرف machine learning دکھانے کے لیے نہیں چاہیے۔',
            faqTwoTitle: 'کون سے platforms supported ہیں؟',
            faqTwoText: 'Windows 10/11 اور macOS Apple Silicon beta۔ اگر automate کرنا ہو تو CLI اور Python library موجود ہیں۔',
            faqThreeTitle: 'کیا یہ مفت ہے؟',
            faqThreeText: 'ہاں۔ Sort Moments MIT license کے تحت free اور open source ہے۔ مشکوک حد تک reasonable، مگر سچ۔',
            footerText: 'Sort Moments مفت، open source، اور Abdullah Khawaja نے بنایا ہے۔',
            currentHomepage: 'موجودہ homepage',
            privacyPolicy: 'پرائیویسی پالیسی',
            archive: 'آرکائیو',
            releases: 'ریلیزز',
            carouselLabel: 'Sort Moments ورک فلو screenshots',
            carouselControlsLabel: 'ورک فلو carousel controls',
            slideProcessingLabel: 'processing stage دکھائیں',
            slideRenameLabel: 'rename stage دکھائیں',
            slideOutputLabel: 'output stage دکھائیں',
            slideReviewLabel: 'review stage دکھائیں',
        },
    };

    const carouselCopy = {
        en: [
            ['Processing', 'Pick a folder. The model works locally. Your Wi-Fi can relax.'],
            ['Rename', 'Rename people before the folders become part of your archive.'],
            ['Output', 'Get ordinary folders back. Revolutionary, if you have used enough photo apps.'],
            ['Review', 'Check the groups, fix the names, move on with your life.'],
        ],
        ur: [
            ['پراسیسنگ', 'فولڈر منتخب کریں۔ ماڈل مقامی طور پر چلتا ہے۔ آپ کا Wi-Fi آرام کر سکتا ہے۔'],
            ['نام بدلیں', 'لوگوں کے نام بدل دیں، اس سے پہلے کہ فولڈرز آپ کے آرکائیو کا حصہ بن جائیں۔'],
            ['آؤٹ پٹ', 'عام فولڈرز واپس ملتے ہیں۔ کافی انقلابی، اگر آپ نے کافی فوٹو ایپس برداشت کی ہیں۔'],
            ['جائزہ', 'گروپس دیکھیں، نام درست کریں، پھر اپنی زندگی کی طرف واپس جائیں۔'],
        ],
    };

    const textBindings = [
        ['.skip-link', 'skip'],
        ['.menu-button .sr-only', 'menu'],
        ['.nav-links a[href="#workflow"]', 'navWorkflow'],
        ['.nav-links a[href="#privacy"]', 'navPrivacy'],
        ['.nav-links a[href="#developers"]', 'navDevelopers'],
        ['.nav-cli-label', 'navCli'],
        ['.nav-links a[href="#demo"]', 'navDemo'],
        ['[data-language-toggle-label]', 'languageToggle'],
        ['.hero .overline', 'heroOverline'],
        ['h1', 'heroTitle'],
        ['.hero-text', 'heroText'],
        ['.hero-stats div:nth-child(1) dt', 'downloads'],
        ['.hero-stats div:nth-child(2) dt', 'visitors'],
        ['[data-download-label="windows"]', 'downloadWindows'],
        ['[data-download-sub="windows"]', 'windowsSub'],
        ['[data-download-label="macos"]', 'downloadMacos'],
        ['[data-download-sub="macos"]', 'macosSub'],
        ['.facts div:nth-child(1) .stat-word', 'downloads'],
        ['#downloadSourceText', 'factsDownloadsNote'],
        ['.facts div:nth-child(2) .stat-word', 'visitors'],
        ['#visitorSourceText', 'factsVisitorsNote'],
        ['.facts div:nth-child(3) strong', 'localDefault'],
        ['.facts div:nth-child(3) > span', 'localDefaultNote'],
        ['.facts div:nth-child(4) strong', 'noAccount'],
        ['.facts div:nth-child(4) > span', 'noAccountNote'],
        ['.workflow .overline', 'workflowOverline'],
        ['#workflow-title', 'workflowTitle'],
        ['.workflow .section-intro > p:not(.overline)', 'workflowText'],
        ['.workflow-item:nth-child(1) h3', 'workflowOneTitle'],
        ['.workflow-item:nth-child(1) p', 'workflowOneText'],
        ['.workflow-item:nth-child(2) h3', 'workflowTwoTitle'],
        ['.workflow-item:nth-child(2) p', 'workflowTwoText'],
        ['.workflow-item:nth-child(3) h3', 'workflowThreeTitle'],
        ['.workflow-item:nth-child(3) p', 'workflowThreeText'],
        ['.workflow-quips .quote-a', 'workflowQuipOne'],
        ['.workflow-quips .quote-b', 'workflowQuipTwo'],
        ['.privacy .overline', 'privacyOverline'],
        ['#privacy-title', 'privacyTitle'],
        ['.privacy-copy p:nth-child(1)', 'privacyTextOne'],
        ['.privacy-copy p:nth-child(2)', 'privacyTextTwo'],
        ['.privacy-quips .quote-a', 'privacyQuipOne'],
        ['.privacy-quips .quote-b', 'privacyQuipTwo'],
        ['.privacy-quips .quote-c', 'privacyQuipThree'],
        ['.privacy-quips .quote-d', 'privacyQuipFour'],
        ['.privacy-quips .quote-e', 'privacyQuipFive'],
        ['.privacy-quips .quote-f', 'privacyQuipSix'],
        ['.developers .overline', 'developersOverline'],
        ['#developers-title', 'developersTitle'],
        ['.developers-quips .quote-a', 'developersQuipOne'],
        ['.developers-quips .quote-b', 'developersQuipTwo'],
        ['.developers-quips .quote-c', 'developersQuipThree'],
        ['.developer-grid > :nth-child(1) .dev-kicker', 'pythonLibrary'],
        ['.developer-grid > :nth-child(1) .dev-status', 'pythonLibrarySoon'],
        ['.developer-grid > :nth-child(1) > strong', 'pythonLibraryText'],
        ['.developer-grid > :nth-child(2) .dev-kicker', 'commandLine'],
        ['.developer-grid > :nth-child(2) > strong', 'commandLineText'],
        ['.demo .overline', 'demoOverline'],
        ['#demo-title', 'demoTitle'],
        ['.demo-quips .quote-a', 'demoQuipOne'],
        ['.demo-quips .quote-b', 'demoQuipTwo'],
        ['.faq .overline', 'answersOverline'],
        ['#faq-title', 'faqTitle'],
        ['.faq-list article:nth-child(1) h3', 'faqOneTitle'],
        ['.faq-list article:nth-child(1) p', 'faqOneText'],
        ['.faq-list article:nth-child(2) h3', 'faqTwoTitle'],
        ['.faq-list article:nth-child(2) p', 'faqTwoText'],
        ['.faq-list article:nth-child(3) h3', 'faqThreeTitle'],
        ['.faq-list article:nth-child(3) p', 'faqThreeText'],
        ['.footer p', 'footerText'],
        ['.footer a[href="/"]', 'currentHomepage'],
        ['.footer a[href="/privacy/"]', 'privacyPolicy'],
        ['.footer a[href="/archive/"]', 'archive'],
        ['.footer a[href*="/releases"]', 'releases'],
    ];

    document.documentElement.classList.add('motion-ready');
    applyLanguage(currentLanguage, false);
    initLanguageToggle();
    initStats();
    initDownloadTracking();
    initCarousel();
    initScrollEffects();
    initStackScroller();

    if (button && nav) {
        button.addEventListener('click', () => {
            setOpen(button.getAttribute('aria-expanded') !== 'true');
        });

        nav.querySelectorAll('a').forEach((link) => {
            link.addEventListener('click', () => setOpen(false));
        });

        document.addEventListener('keydown', (event) => {
            if (event.key === 'Escape') setOpen(false);
        });
    }

    function setOpen(open) {
        button.setAttribute('aria-expanded', String(open));
        nav.classList.toggle('is-open', open);
    }

    function initLanguageToggle() {
        if (!languageToggle) return;
        languageToggle.addEventListener('click', () => {
            const next = currentLanguage === 'ur' ? 'en' : 'ur';
            applyLanguage(next, true);
        });
    }

    function getInitialLanguage() {
        const params = new URLSearchParams(window.location.search);
        const paramLanguage = params.get('lang');
        if (paramLanguage === 'ur' || params.has('urdu')) return 'ur';
        if (paramLanguage === 'en') return 'en';

        try {
            const stored = window.localStorage.getItem('sortmoments-language');
            if (stored === 'ur' || stored === 'en') return stored;
        } catch (error) {
            // Storage is optional. Keep the page usable if it is blocked.
        }
        return 'en';
    }

    function applyLanguage(language, persist) {
        currentLanguage = language === 'ur' ? 'ur' : 'en';
        const strings = copy[currentLanguage];
        document.documentElement.lang = currentLanguage;
        document.documentElement.dir = currentLanguage === 'ur' ? 'rtl' : 'ltr';

        textBindings.forEach(([selector, key]) => setText(selector, strings[key]));
        setAttr('nav', 'aria-label', strings.primaryNav);
        setAttr('.facts', 'aria-label', strings.factsLabel);
        setAttr('.carousel-frame', 'aria-label', strings.carouselLabel);
        setAttr('.carousel-controls', 'aria-label', strings.carouselControlsLabel);

        if (languageToggle) {
            languageToggle.setAttribute('aria-label', strings.languageButtonLabel);
            languageToggle.setAttribute('aria-pressed', String(currentLanguage === 'ur'));
        }

        updateCarouselControls();
        if (refreshCarouselLanguage) refreshCarouselLanguage();
        if (refreshMobileStacks) refreshMobileStacks();
        updateStatsSourceCopy();

        if (persist) {
            try {
                window.localStorage.setItem('sortmoments-language', currentLanguage);
            } catch (error) {
                // Storage is optional. The toggle still works for this page view.
            }
        }
    }

    function setText(selector, value) {
        const element = document.querySelector(selector);
        if (!element || typeof value !== 'string') return;
        element.textContent = value;
    }

    function setAttr(selector, attribute, value) {
        const element = document.querySelector(selector);
        if (!element || typeof value !== 'string') return;
        element.setAttribute(attribute, value);
    }

    function updateCarouselControls() {
        const labels = [
            copy[currentLanguage].slideProcessingLabel,
            copy[currentLanguage].slideRenameLabel,
            copy[currentLanguage].slideOutputLabel,
            copy[currentLanguage].slideReviewLabel,
        ];
        document.querySelectorAll('.carousel-controls button').forEach((control, index) => {
            control.setAttribute('aria-label', labels[index] || labels[0]);
        });
    }

    function initCarousel() {
        const root = document.querySelector('[data-carousel]');
        if (!root) return;

        const slides = Array.from(root.querySelectorAll('.carousel-slide'));
        const controls = Array.from(root.querySelectorAll('.carousel-controls button'));
        const title = document.getElementById('carouselTitle');
        const caption = document.getElementById('carouselCaption');
        let activeIndex = 0;
        let timer = null;

        function setSlide(index) {
            activeIndex = (index + slides.length) % slides.length;

            slides.forEach((slide, slideIndex) => {
                slide.classList.toggle('is-active', slideIndex === activeIndex);
            });

            controls.forEach((control, controlIndex) => {
                const isActive = controlIndex === activeIndex;
                control.classList.toggle('is-active', isActive);
                control.setAttribute('aria-pressed', String(isActive));
            });

            const localized = carouselCopy[currentLanguage][activeIndex] || carouselCopy.en[activeIndex];
            if (title) title.textContent = localized[0];
            if (caption) caption.textContent = localized[1];
        }

        function start() {
            if (reduceMotion || timer) return;
            timer = window.setInterval(() => setSlide(activeIndex + 1), 3600);
        }

        function stop() {
            if (!timer) return;
            window.clearInterval(timer);
            timer = null;
        }

        controls.forEach((control, index) => {
            control.addEventListener('click', () => {
                stop();
                setSlide(index);
                start();
            });
        });

        root.addEventListener('mouseenter', stop);
        root.addEventListener('mouseleave', start);
        root.addEventListener('focusin', stop);
        root.addEventListener('focusout', start);

        refreshCarouselLanguage = () => setSlide(activeIndex);
        updateCarouselControls();
        setSlide(0);
        start();
    }

    function initScrollEffects() {
        const revealItems = Array.from(document.querySelectorAll('.reveal-seq'));

        if (reduceMotion) {
            revealItems.forEach((item) => item.classList.add('is-revealed'));
            updateScrollVeil();
            return;
        }

        revealItems.forEach((item) => {
            let delay = 0;
            if (item.classList.contains('reveal-right')) delay += 110;
            if (item.classList.contains('reveal-up')) delay += 210;
            if (item.classList.contains('pixel-life')) delay += 120;
            item.style.setProperty('--reveal-delay', `${delay}ms`);
        });

        if ('IntersectionObserver' in window) {
            const observer = new IntersectionObserver((entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    entry.target.classList.add('is-revealed');
                    observer.unobserve(entry.target);
                });
            }, { threshold: 0.16, rootMargin: '0px 0px -8% 0px' });

            revealItems.forEach((item) => observer.observe(item));
        } else {
            revealItems.forEach((item) => item.classList.add('is-revealed'));
        }

        let ticking = false;
        const onScroll = () => {
            if (ticking) return;
            ticking = true;
            window.requestAnimationFrame(() => {
                updateScrollVeil();
                ticking = false;
            });
        };

        onScroll();
        window.addEventListener('scroll', onScroll, { passive: true });
        window.addEventListener('resize', onScroll);
    }

    function updateScrollVeil() {
        const scrollable = Math.max(1, document.documentElement.scrollHeight - window.innerHeight);
        const progress = Math.min(1, Math.max(0, (window.scrollY || 0) / scrollable));
        const visible = Math.min(0.72, Math.max(0, progress * 1.45));
        document.documentElement.style.setProperty('--scroll-progress', progress.toFixed(4));
        document.documentElement.style.setProperty('--veil-opacity', visible.toFixed(3));
    }

    function initStackScroller() {
        const mobileStack = window.matchMedia('(max-width: 720px)');
        const stackDefinitions = [
            { selector: '.hero' },
            { selector: '#workflow', quipKey: 'workflowQuipOne' },
            { selector: '#privacy', quipKey: 'privacyQuipOne' },
            { selector: '#developers', quipKey: 'developersQuipOne' },
            { selector: '#demo', quipKey: 'demoQuipOne' },
        ];
        let stacks = [];
        let locked = false;
        let touchStartY = 0;
        let touchConsumed = false;
        let activeFrame = 0;

        function refreshStacks() {
            stacks = stackDefinitions
                .flatMap((definition) => {
                    return Array.from(document.querySelectorAll(definition.selector)).map((element) => ({ element, definition }));
                })
                .filter((entry, index, list) => list.findIndex((item) => item.element === entry.element) === index)
                .filter((entry) => entry.element.getBoundingClientRect().height > 24);
            stacks.forEach(({ element, definition }, index) => prepareStack(element, definition, index));
            updateActiveStack();
        }

        function isInteractiveTarget(target) {
            if (!target || typeof target.closest !== 'function') return false;
            return Boolean(target.closest('a, button, input, textarea, select, video, .nav-links, .carousel-controls'));
        }

        function stackOffset() {
            const navRect = document.querySelector('.site-header')?.getBoundingClientRect();
            return Math.round((navRect?.bottom || 82) + 12);
        }

        function nearestStackIndex() {
            const offset = stackOffset();
            let nearest = 0;
            let nearestDistance = Number.POSITIVE_INFINITY;
            stacks.forEach(({ element }, index) => {
                const distance = Math.abs(element.getBoundingClientRect().top - offset);
                if (distance < nearestDistance) {
                    nearest = index;
                    nearestDistance = distance;
                }
            });
            return nearest;
        }

        function updateActiveStack() {
            if (!stacks.length) return;
            const active = mobileStack.matches ? nearestStackIndex() : -1;
            stacks.forEach(({ element }, index) => {
                element.classList.toggle('is-active-stack', index === active);
            });
        }

        function queueActiveUpdate() {
            if (activeFrame) return;
            activeFrame = window.requestAnimationFrame(() => {
                activeFrame = 0;
                updateActiveStack();
            });
        }

        function go(direction) {
            if (!mobileStack.matches || reduceMotion || locked) return;
            if (!stacks.length) refreshStacks();
            if (!stacks.length) return;

            const current = nearestStackIndex();
            const next = Math.max(0, Math.min(stacks.length - 1, current + direction));
            const target = stacks[next]?.element;
            if (!target || next === current) return;

            locked = true;
            const top = target.getBoundingClientRect().top + window.scrollY - stackOffset();
            window.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
            queueActiveUpdate();
            window.setTimeout(() => {
                locked = false;
                updateActiveStack();
            }, 620);
        }

        refreshStacks();
        refreshMobileStacks = refreshStacks;
        window.addEventListener('resize', () => {
            refreshStacks();
            queueActiveUpdate();
        });
        window.addEventListener('load', () => {
            refreshStacks();
            queueActiveUpdate();
        });
        window.addEventListener('scroll', queueActiveUpdate, { passive: true });

        window.addEventListener('wheel', (event) => {
            if (!mobileStack.matches || Math.abs(event.deltaY) < 12 || isInteractiveTarget(event.target)) return;
            event.preventDefault();
            go(event.deltaY > 0 ? 1 : -1);
        }, { passive: false });

        window.addEventListener('touchstart', (event) => {
            if (!mobileStack.matches || isInteractiveTarget(event.target)) return;
            touchStartY = event.touches[0]?.clientY || 0;
            touchConsumed = false;
        }, { passive: true });

        window.addEventListener('touchmove', (event) => {
            if (!mobileStack.matches || touchConsumed || isInteractiveTarget(event.target)) return;
            const currentY = event.touches[0]?.clientY || touchStartY;
            const delta = touchStartY - currentY;
            if (Math.abs(delta) < 46) return;
            event.preventDefault();
            touchConsumed = true;
            go(delta > 0 ? 1 : -1);
        }, { passive: false });

        window.addEventListener('touchend', () => {
            touchStartY = 0;
            touchConsumed = false;
        }, { passive: true });

        function prepareStack(element, definition, index) {
            element.classList.add('mobile-stack');
            element.classList.toggle('is-mobile-stack-alt', index % 2 === 1);
            element.style.setProperty('--stack-index', String(index));
            let quip = element.querySelector(':scope > .mobile-quip');
            if (!definition.quipKey) {
                if (quip) quip.remove();
                return;
            }
            if (!quip) {
                quip = document.createElement('span');
                quip.className = 'mobile-quip hand-note';
                quip.setAttribute('aria-hidden', 'true');
                element.appendChild(quip);
            }
            quip.textContent = copy[currentLanguage][definition.quipKey] || copy.en[definition.quipKey];
        }
    }

    async function initStats() {
        const downloadHero = document.getElementById('downloadCountAlt');
        const visitorHero = document.getElementById('visitorCountAlt');
        const downloadFact = document.getElementById('downloadCountFact');
        const visitorFact = document.getElementById('visitorCountFact');
        const downloadSource = document.getElementById('downloadSourceText');
        const visitorSource = document.getElementById('visitorSourceText');

        if (!downloadHero || !visitorHero) return;

        const fallback = {
            downloads: downloadHero.dataset.fallback || '300+',
            visitors: visitorHero.dataset.fallback || '2.2k+',
        };

        function setStats(downloads, visitors) {
            const formattedDownloads = formatCount(downloads, fallback.downloads);
            const formattedVisitors = formatCount(visitors, fallback.visitors);
            const proofDownloads = formatDownloadProofCount(downloads, '300+');
            const proofVisitors = formatVisitorProofCount(visitors, fallback.visitors);
            downloadHero.textContent = formattedDownloads;
            visitorHero.textContent = formattedVisitors;
            if (downloadFact) downloadFact.textContent = proofDownloads;
            if (visitorFact) visitorFact.textContent = proofVisitors;
            latestStats = {
                downloads: parseCount(downloads),
                visitors: parseCount(visitors),
                formattedDownloads,
                formattedVisitors,
                proofDownloads,
                proofVisitors,
            };
            updateStatsSourceCopy();
        }

        try {
            const data = await fetchAnalytics();
            setStats(data.downloads, data.unique_visitors);
        } catch (error) {
            setStats(fallback.downloads, fallback.visitors);
        }
    }

    async function fetchAnalytics() {
        const endpoints = isStaticLocalPreview()
            ? [
                'https://sortmoments.com/api/analytics?track=0',
            ]
            : ['/api/analytics'];

        let lastError;
        for (const endpoint of endpoints) {
            try {
                const response = await fetch(endpoint, { cache: 'no-store' });
                if (!response.ok) throw new Error(`Stats returned ${response.status}`);
                return await response.json();
            } catch (error) {
                lastError = error;
            }
        }
        throw lastError || new Error('Stats unavailable');
    }

    function formatCount(value, fallback) {
        const number = parseCount(value);
        if (!Number.isFinite(number) || number <= 0) return fallback;
        if (number <= 9999) return number.toLocaleString();
        return formatCompactCount(number);
    }

    function parseCount(value) {
        if (typeof value === 'number') return value;
        if (typeof value !== 'string') return Number.NaN;
        const normalized = value.trim().replace(/,/g, '');
        if (!/^\d+(\.\d+)?$/.test(normalized)) return Number.NaN;
        return Number(normalized);
    }

    function formatCompactCount(number) {
        const units = [
            { threshold: 1000000, value: 1000000, suffix: 'm' },
            { threshold: 10000, value: 1000, suffix: 'k' },
        ];
        const unit = units.find((item) => number >= item.threshold);
        if (!unit) return number.toLocaleString();
        const floored = Math.floor((number / unit.value) * 10) / 10;
        const label = Number.isInteger(floored) ? String(floored) : floored.toFixed(1);
        return `${label}${unit.suffix}+`;
    }

    function formatDownloadProofCount(value, fallback) {
        const number = parseCount(value);
        if (!Number.isFinite(number) || number <= 0) return fallback;
        if (number < 100) return number.toLocaleString();
        if (number <= 9999) return `${Math.floor(number / 100) * 100}+`;
        return formatCompactCount(number);
    }

    function formatVisitorProofCount(value, fallback) {
        const number = parseCount(value);
        if (!Number.isFinite(number) || number <= 0) return fallback;
        if (number < 1000) return number.toLocaleString();
        return `${Math.floor(number / 1000)}K+`;
    }

    function incrementDownloadStats(count) {
        const downloadHero = document.getElementById('downloadCountAlt');
        const downloadFact = document.getElementById('downloadCountFact');
        const currentDisplayedCount = parseCount(downloadHero?.textContent);
        const nextCount = Number.isFinite(count)
            ? count
            : (Number.isFinite(latestStats?.downloads)
                ? latestStats.downloads + 1
                : currentDisplayedCount + 1);

        if (!Number.isFinite(nextCount) || nextCount <= 0) return;

        const formattedDownloads = formatCount(nextCount, downloadHero?.dataset.fallback || '300+');
        const proofDownloads = formatDownloadProofCount(nextCount, '300+');
        if (downloadHero) downloadHero.textContent = formattedDownloads;
        if (downloadFact) downloadFact.textContent = proofDownloads;
        latestStats = {
            ...(latestStats || {}),
            downloads: nextCount,
            formattedDownloads,
            proofDownloads,
        };
        updateStatsSourceCopy();
    }

    function initDownloadTracking() {
        const links = document.querySelectorAll('.hero-actions a[href*="/releases/download/"]');
        links.forEach((link) => {
            link.addEventListener('click', () => {
                incrementDownloadStats();
                const endpoint = isStaticLocalPreview()
                    ? 'https://sortmoments.com/api/counter/increment'
                    : '/api/counter/increment';
                fetch(endpoint, {
                    method: 'GET',
                    cache: 'no-store',
                    keepalive: true,
                })
                    .then((response) => (response.ok ? response.json() : null))
                    .then((data) => {
                        const serverCount = Number(data?.count);
                        if (Number.isFinite(serverCount)) incrementDownloadStats(serverCount);
                    })
                    .catch(() => {});
            }, { passive: true });
        });
    }

    function updateStatsSourceCopy() {
        if (!latestStats) return;
        const downloadSource = document.getElementById('downloadSourceText');
        const visitorSource = document.getElementById('visitorSourceText');
        if (currentLanguage === 'ur') {
            if (downloadSource) {
                downloadSource.textContent = copy.ur.factsDownloadsNote;
            }
            if (visitorSource) {
                visitorSource.textContent = copy.ur.factsVisitorsNote;
            }
            return;
        }
        if (downloadSource) {
            downloadSource.textContent = copy.en.factsDownloadsNote;
        }
        if (visitorSource) {
            visitorSource.textContent = copy.en.factsVisitorsNote;
        }
    }

    function isStaticLocalPreview() {
        return (
            (location.hostname === '127.0.0.1' || location.hostname === 'localhost')
            && location.port === '8000'
        );
    }
}());
