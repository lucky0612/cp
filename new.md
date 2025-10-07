/* static/css/animation.css */
/* Base & Keyframe Animations */

@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeOut {
    from { opacity: 1; }
    to { opacity: 0; }
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: translateY(-30px);
    }
    60% {
        opacity: 1;
        transform: translateY(5px);
    }
    100% {
        transform: translateY(0);
    }
}

@keyframes slideInFromLeft {
    from {
        opacity: 0;
        transform: translateX(-20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInFromRight {
    from {
        opacity: 0;
        transform: translateX(20px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideOutToRight {
    from {
        opacity: 1;
        transform: translateX(0);
    }
    to {
        opacity: 0;
        transform: translateX(20px);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0.8;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes pulse {
    0%, 100% {
        transform: scale(1);
    }
    50% {
        transform: scale(1.05);
    }
}

@keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
        transform: translateY(0);
    }
    40% {
        transform: translateY(-10px);
    }
    60% {
        transform: translateY(-5px);
    }
    90% {
        transform: translateY(-2px);
    }
}

@keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}

@keyframes shimmer {
    0% {
        background-position: -200px 0;
    }
    100% {
        background-position: calc(200px + 100%) 0;
    }
}

@keyframes slideOutRight {
    from {
        transform: translateX(0);
        opacity: 1;
    }
    to {
        transform: translateX(100%);
        opacity: 0;
    }
}

@keyframes slideInRight {
    from {
        transform: translateX(100%);
        opacity: 0;
    }
    to {
        transform: translateX(0);
        opacity: 1;
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 5px rgba(74, 144, 226, 0.3);
    }
    50% {
        box-shadow: 0 0 20px rgba(74, 144, 226, 0.6);
    }
}

@keyframes typewriter {
    from { width: 0; }
    to { width: 100%; }
}

@keyframes blinkCursor {
    from, to { border-color: transparent; }
    50% { border-color: var(--primary-blue); }
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Page Load Animations */
.intro-section {
    animation: fadeInDown 0.8s ease-out;
}

.form-card {
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

.results-section.show {
    animation: fadeInUp 0.8s ease-out;
}

/* Component Animations */
.scenario-info {
    animation: scaleIn 0.3s ease-out;
}

.summary-card:nth-child(1) { animation-delay: 0.1s; }
.summary-card:nth-child(2) { animation-delay: 0.2s; }
.summary-card:nth-child(3) { animation-delay: 0.3s; }

.table-container {
    animation: fadeInUp 0.6s ease-out;
}

.table-container tbody tr:nth-child(1) { animation-delay: 0.1s; }
.table-container tbody tr:nth-child(2) { animation-delay: 0.2s; }

/* Interactive Animations */
.execute-btn {
    overflow: hidden;
    position: relative;
}

.execute-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -80%;
    width: 30%;
    height: 100%;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.3), transparent);
    transition: left 0.5s;
}

.execute-btn:hover::before {
    left: 100%;
}

.btn-secondary {
    position: relative;
    overflow: hidden;
}

.btn-secondary:after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 5px;
    height: 5px;
    background: rgba(0,0,0,0.05);
    border-radius: 50%;
    transform: translate(-50%, -50%) scale(1);
    transition: width 0.6s, height 0.6s;
}

.btn-secondary:active:after {
    width: 300px;
    height: 300px;
}

/* Status Badge Animations */
.status-badge {
    position: relative;
    overflow: hidden;
}

.status-badge-match {
    animation: pulse 2s infinite;
}

.status-badge-mismatch {
    animation: pulse 1.5s infinite;
}

/* Form Input Animations */
.form-group:focus-within,
.form-group.input-has-focus {
    animation: glow 1s ease-in-out infinite;
}

.input-info {
    transform: translateY(-10px);
    opacity: 0;
    transition: all 0.3s;
}

.form-group:focus-within .input-info,
.form-group.input-has-focus .input-info {
    transform: translateY(0);
    opacity: 1;
}

/* Connection Status Animation */
.connection-status.connected {
    animation: pulse 2s infinite;
}

.connection-status.disconnected {
    animation: bounce 1s infinite;
}

/* Table Row Animations */
.comparison-table tbody tr {
    transform: translateX(-20px);
    opacity: 0;
    animation: slideInFromLeft 0.5s ease-out forwards;
}
.comparison-table tbody tr:nth-child(1) { animation-delay: 0.1s; }
.comparison-table tbody tr:nth-child(2) { animation-delay: 0.2s; }
.comparison-table tbody tr:nth-child(3) { animation-delay: 0.3s; }
.comparison-table tbody tr:nth-child(4) { animation-delay: 0.4s; }
.comparison-table tbody tr:nth-child(5) { animation-delay: 0.5s; }

/* Action Icon Animations */
.action-icon {
    transition: all 0.3s cubic-bezier(0.42, -0.55, 0.58, 1.55);
}

.action-icon:hover {
    animation: bounce 0.8s ease;
}

/* Card Hover Effects */
.summary-card {
    transition: all 0.3s ease;
}
.summary-card:hover {
    animation: none;
    transform: translateY(-4px) scale(1.03);
}

/* Loading Animations */
.loading-dots::after {
    content: '...';
    animation: typewriter 1.5s steps(3) infinite;
}

/* Shimmer Effect for Loading States */
.shimmer {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: shimmer 1.5s infinite;
}

/* Scenario Info Animation */
.scenario-info {
    animation: slideInFromLeft 0.6s ease-out;
}
.scenario-info .info-item {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.6s ease-out forwards;
}

.scenario-info .action-btn:nth-child(1) { animation-delay: 0.1s; }
.scenario-info .action-btn:nth-child(2) { animation-delay: 0.2s; }
.scenario-info .action-btn:nth-child(3) { animation-delay: 0.3s; }

/* Toast Notification Animations */
.toast {
    animation: slideInRight 0.3s ease-out;
}
.toast.removing {
    animation: slideOutRight 0.3s ease-out;
}

/* Progress Bar Animation */
.progress-bar-inner {
    width: 0%;
    background: var(--primary-blue);
    height: 100%;
    transition: width 0.5s ease;
}
.progress-bar.loading .progress-bar-inner {
    animation: none 1.5s infinite;
}

/* Number Counter Animation */
.card-number {
    animation: fadeIn 0.8s ease-out 0.5s forwards;
    opacity: 0;
}
.card-number.counting {
    animation: bounce 0.8s ease-in-out;
}

/* Stagger Animation for Grid Items */
.form-row.growth {
    transform: translateY(20px);
    opacity: 0;
    animation: fadeInUp 0.8s ease-out forwards;
}
.form-row.growth:nth-child(1) { animation-delay: 0.1s; }
.form-row.growth:nth-child(2) { animation-delay: 0.2s; }
.form-row.growth:nth-child(3) { animation-delay: 0.3s; }
.form-row.growth:nth-child(4) { animation-delay: 0.4s; }

/* Header Animation */
.logo {
    animation: fadeInDown 0.8s ease-out 0.2s both;
}
.connection-status {
    animation: fadeInDown 0.8s ease-out 0.4s both;
}

/* Elastic Scale Animation */
@keyframes elasticScale {
    0% { transform: scale(1); }
    30% { transform: scale(1.1); }
    60% { transform: scale(0.95); }
    80% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.execute-btn:active {
    animation: elasticScale 0.6s ease-out;
}

/* Floating Animation */
@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-5px);
    }
}
.floating {
    animation: float 3s ease-in-out infinite;
}

/* Attention Seeking Animations */
@keyframes shake {
    0%, 100% { transform: translateX(0); }
    10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
    20%, 40%, 60%, 80% { transform: translateX(5px); }
}
.shake {
    animation: shake 0.6s ease-in-out;
}

/* Success Animation */
@keyframes successPulse {
    0% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(39, 174, 96, 0.7);
    }
    70% {
        transform: scale(1);
        box-shadow: 0 0 10px 10px rgba(39, 174, 96, 0);
    }
    100% {
        transform: scale(1);
        box-shadow: 0 0 0 0 rgba(39, 174, 96, 0);
    }
}
.success-animation {
    animation: successPulse 0.8s ease-out;
}

/* Error Animation */
@keyframes errorShake {
    0%, 100% {
        transform: translateX(0);
    }
    25% {
        transform: translateX(-8px);
    }
    75% {
        transform: translateX(8px);
    }
}
.error-animation {
    animation: errorShake 0.5s ease-out;
}

/* Reduced Motion Support */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
        scroll-behavior: auto !important;
    }
}

/* Animation Utility Classes */
.animate-fade-in { animation: fadeIn 0.5s ease-out; }
.animate-fade-out { animation: fadeOut 0.5s ease-out; }
.animate-slide-in-left { animation: slideInFromLeft 0.5s ease-out; }
.animate-slide-in-right { animation: slideInFromRight 0.5s ease-out; }
.animate-bounce-in { animation: bounceIn 0.6s ease-out; }
.animate-scale-in { animation: scaleIn 0.3s ease-out; }
.animate-pulse { animation: pulse 1.5s ease-in-out infinite; }
.animate-shimmer { animation: shimmer 1.5s ease-in-out infinite; }
.animate-spin { animation: spin 1s linear infinite; }
.animate-glow { animation: glow 2s ease-in-out infinite; }

/* Animation Speeds */
.delay-100 { animation-delay: 0.1s; }
.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-400 { animation-delay: 0.4s; }
.delay-500 { animation-delay: 0.5s; }

.animation-fast { animation-duration: 0.3s; }
.animation-slow { animation-duration: 2s; }
















/* static/css/style.css */

/* --- Base Styles & Variables --- */
*,
*::before,
*::after {
    box-sizing: border-box;
}

:root {
    /* Color Palette - Using exact provided colors */
    --primary-color: #1a73e8;
    --primary-light: #669df6;
    --primary-dark: #174ea6;
    --accent-color: #fbbc05;
    --success-color: #34a853;
    --error-color: #ea4335;
    --warning-color: #f29900;
    --bg-color: #1e1e1e; /* Dark theme bg */

    /* Neutral Colors */
    --white: #ffffff;
    --gray-100: #f8f9fa;
    --gray-200: #e9ecef;
    --gray-300: #dee2e6;
    --gray-400: #ced4da;
    --gray-500: #adb5bd;
    --gray-600: #6c757d;
    --gray-700: #495057;
    --gray-800: #343a40;
    --gray-900: #212529;

    /* Dark Background Variants */
    --bg-card: #2a2a2e;
    --bg-secondary: #3c4043;
    --bg-darker: #121212;

    /* Gradients */
    --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #66bb6a 100%);
    --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #ef5350 100%);
    --gradient-warning: linear-gradient(135deg, var(--warning-color) 0%, #ffca28 100%);

    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    --font-size-sm: 0.875rem; /* 14px */
    --font-size-base: 1rem; /* 16px */
    --font-size-md: 1.125rem; /* 18px */
    --font-size-lg: 1.5rem; /* 24px */
    --font-size-xl: 2.25rem; /* 36px */
    --font-size-xxl: 2.5rem; /* 40px */

    /* Spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-xxl: 3rem;

    /* Border Radius */
    --radius-sm: 0.25rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    --radius-full: 9999px;

    /* Shadows - Enhanced for depth */
    --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.1);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.3), 0 10px 10px -5px rgba(0, 0, 0, 0.15);

    /* Transitions */
    --transition-fast: 0.15s ease-in-out;
    --transition-normal: 0.3s ease-in-out;
    --transition-slow: 0.5s ease-in-out;
}

/* --- Base Typography --- */
body {
    font-family: var(--font-family);
    line-height: 1.6;
    color: var(--gray-200);
    background-color: var(--bg-color);
}

h1, h2, h3, h4, h5, h6 {
    color: var(--white);
    font-weight: 700;
}

/* --- Layout --- */
.container {
    width: 90%;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

/* --- Header --- */
.page-header {
    background: var(--gradient-primary);
    padding: var(--spacing-lg) 0;
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 2px solid var(--accent-color);
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

.logo {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.logo i {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.logo h1 {
    font-size: var(--font-size-lg);
    font-weight: 700;
    color: var(--white);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
}

.connection-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    background: rgba(255, 255, 255, 0.15);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-full);
    border: 1px solid rgba(255, 255, 255, 0.3);
    box-shadow: var(--shadow-sm);
}

.status-dot,
.connection-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    background: var(--warning-color);
    border: 2px solid var(--white);
    box-shadow: 0 0 8px currentColor;
}

.status-dot.connected,
.connection-dot.connected {
    background: var(--success-color);
}

.status-dot.disconnected,
.connection-dot.disconnected,
.connection-dot.status-dot-degraded {
    background: var(--error-color);
}

/* --- Main Content --- */
.main-content {
    padding: var(--spacing-xxl) 0;
    min-height: calc(100vh - 100px);
    max-width: 1400px;
    margin: 0 auto;
    padding-left: var(--spacing-lg);
    padding-right: var(--spacing-lg);
}

/* --- Welcome Section --- */
.intro-section {
    text-align: center;
    margin-bottom: var(--spacing-xxl);
}

.intro-content h1 {
    font-size: var(--font-size-xxl);
    font-weight: 700;
    margin-bottom: var(--spacing-md);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.intro-content p {
    font-size: var(--font-size-lg);
    max-width: 600px;
    margin: 0 auto;
}

/* --- NEW: Environment Selection Section --- */
.environment-section {
    margin-bottom: var(--spacing-xxl);
}

.environment-card {
    background: var(--bg-card);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(10px);
}

.environment-card h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
    margin-bottom: var(--spacing-lg);
    text-align: center;
}

/* --- NEW: Reconciliation Type Section --- */
.recon-type-section {
    margin-bottom: var(--spacing-xxl);
}

.recon-type-card {
    background: var(--bg-card);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(10px);
}

.recon-type-card h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
    margin-bottom: var(--spacing-lg);
    text-align: center;
}

.tab-container {
    display: flex;
    gap: var(--spacing-md);
    justify-content: center;
    flex-wrap: wrap;
}

.tab-button {
    background: var(--bg-darker);
    color: var(--white);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-md) var(--spacing-xl);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-md);
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.tab-button:hover {
    border-color: var(--accent-color);
    background: var(--primary-dark);
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
}

.tab-button.active {
    background: var(--gradient-primary);
    border-color: var(--accent-color);
    box-shadow: var(--shadow-lg);
}

.tab-button i {
    font-size: var(--font-size-md);
}

/* --- Search/Reconciliation Section --- */
.search-section,
.reconciliation-section {
    margin-bottom: var(--spacing-xxl);
}

.search-card,
.recon-header {
    text-align: center;
    margin-bottom: var(--spacing-lg);
}

.search-header h2,
.recon-header h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
}

.search-header p,
.recon-header p {
    color: var(--gray-300);
    font-size: var(--font-size-md);
}

/* --- Form Grid --- */
#reconcile-form {
    background: var(--bg-card);
    border: 1px solid var(--gray-800);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-lg);
    backdrop-filter: blur(10px);
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg) var(--spacing-xl);
}

.form-row {
    margin-bottom: var(--spacing-lg);
}

.form-row-centered {
    display: flex;
    justify-content: center;
}

.form-row-centered .form-group {
    width: 100%;
    max-width: 500px;
}

/* NEW: Split row for GFID and GUS_ID side by side */
.form-row-split {
    display: flex;
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-lg);
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.form-group-half {
    flex: 1;
}

.form-label {
    font-size: var(--font-size-sm);
    color: var(--white);
}

.form-group label i {
    color: var(--primary-light);
    margin-right: var(--spacing-sm);
}

.form-input {
    padding: var(--spacing-md);
    border: 1px solid var(--primary-color);
    border-radius: var(--radius-md);
    font-size: var(--font-size-base);
    color: var(--white);
    background: var(--bg-darker);
    transition: var(--transition-normal);
    box-shadow: var(--shadow-sm);
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: var(--accent-color);
    box-shadow: 0 0 8px 0 rgba(0, 188, 212, 0.3);
    transform: translateY(-2px);
}

.form-input:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    background: rgba(0, 0, 0, 0.3);
}

.form-group input::placeholder {
    color: var(--gray-600);
}

.input-info,
.form-hint {
    font-size: var(--font-size-sm);
    color: var(--gray-400);
}

select.form-input {
    cursor: pointer;
}

select.form-input option:disabled {
    color: var(--gray-600);
}

.submit-container {
    text-align: center;
    padding-top: var(--spacing-xl);
}

/* --- Buttons --- */
.btn {
    --btn-primary: var(--primary-color);
    background: var(--gradient-primary);
    color: var(--white);
    padding: var(--spacing-md) var(--spacing-xxl);
    border: 2px solid transparent;
    border-radius: var(--radius-full);
    font-size: var(--font-size-lg);
    font-weight: 600;
    cursor: pointer;
    display: inline-flex;
    justify-content: center;
    align-items: center;
    white-space: nowrap;
    transition: var(--transition-normal);
    text-shadow: 0 1px 2px rgba(0,0,0,0.2);
    margin: 0 auto;
    gap: var(--spacing-sm);
}

.btn:hover:not(:disabled) {
    box-shadow: var(--shadow-md);
    border-color: var(--accent-color);
    transform: translateY(-2px);
}

.btn:primary:active {
    transform: translateY(0px);
}

.btn:primary:disabled,
.btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--white);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-full);
    font-size: var(--font-size-md);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    box-shadow: var(--shadow-md);
    gap: var(--spacing-sm);
}

.btn-secondary:hover {
    border-color: var(--accent-color);
    background: var(--primary-light);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.table-section {
    padding: var(--spacing-xl) var(--spacing-sm);
    border-radius: var(--radius-lg);
    width: auto;
    margin: 0;
}

/* --- Results Section --- */
.results-section,
.result-section {
    margin-bottom: var(--spacing-xxl);
    display: none;
}

.results-section.show,
.result-section.show {
    display: block;
    animation: fadeInUp 0.8s ease-out;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xl);
}

.results-header h2 {
    font-size: var(--font-size-xl);
    color: var(--accent-color);
}

.results-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.results-actions {
    display: flex;
    gap: var(--spacing-md);
}

/* --- Summary Cards --- */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
}

.summary-card {
    background: var(--bg-card);
    border-radius: var(--radius-lg);
    padding: var(--spacing-lg);
    border-left: 5px solid var(--primary-color);
    transition: var(--transition-normal);
    border: 2px solid var(--primary-dark);
    text-align: center;
}

.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--accent-color);
}

.summary-card.total {
    border-left-color: var(--accent-color);
}

.summary-card.matches {
    border-left-color: var(--success-color);
}

.summary-card.mismatches {
    border-left-color: var(--error-color);
}

.summary-card.missing {
    border-left-color: var(--warning-color);
}

.card-number {
    font-size: var(--font-size-xl);
    font-weight: 700;
    margin-bottom: var(--spacing-sm);
}

.total .card-number {
    color: var(--accent-color);
}

.matches .card-number {
    color: var(--success-color);
}

.mismatches .card-number {
    color: var(--error-color);
}

.missing .card-number {
    color: var(--warning-color);
}

.card-text {
    font-size: var(--font-size-md);
    color: var(--gray-300);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* --- Scenario Info --- */
.scenario-hidden {
    display: none;
}

.scenario-info,
.scenario-card {
    background: var(--bg-card);
    padding: var(--spacing-xl);
    border-radius: var(--radius-xl);
    margin-top: var(--spacing-xl);
    border: 2px solid var(--accent-color);
    box-shadow: var(--shadow-lg);
}

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
}

.scenario-title {
    font-size: var(--font-size-lg);
    color: var(--white);
}

.severity-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-full);
    font-size: var(--font-size-sm);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.severity-badge.low,
.severity-badge.severity-low {
    background: var(--success-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--success-color);
}

.severity-badge.medium,
.severity-badge.severity-medium {
    background: var(--warning-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--warning-color);
}

.severity-badge.high,
.severity-badge.severity-high {
    background: var(--error-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--error-color);
}

.scenario-content p {
    color: var(--gray-300);
    font-size: var(--font-size-base);
}

.scenario-content strong {
    color: var(--white);
}

.recommended-actions-h5 {
    margin-top: var(--spacing-lg);
    margin-bottom: var(--spacing-md);
    color: var(--accent-color);
}

.actions-buttons {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-md);
}

.action-btn {
    background: var(--gradient-primary);
    color: var(--white);
    padding: var(--spacing-sm) var(--spacing-lg);
    border: none;
    border-radius: var(--radius-lg);
    font-weight: 500;
    font-size: var(--font-size-md);
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-right: var(--spacing-md);
    border: 1px solid var(--accent-color);
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-md);
    border-color: var(--white);
}

/* --- Comparison Table --- */
.comparison-table {
    margin-top: var(--spacing-xxl);
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.single-comparison-table {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    overflow: hidden;
    border: 2px solid var(--primary-color);
}

.table-section {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    overflow: hidden;
    border: 2px solid var(--primary-color);
}

.table-header {
    background: var(--gradient-primary);
    padding: var(--spacing-md);
    text-align: center;
    border-bottom: 2px solid var(--accent-color);
}

.table-header h3 {
    color: var(--white);
    font-size: var(--font-size-lg);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    margin: 0;
}

.table-header th {
    margin-right: var(--spacing-sm);
    color: var(--white);
}

.table-container {
    overflow-x: auto;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    border-color: var(--primary-color);
}

.comparison-table th {
    background: var(--primary-dark);
    padding: var(--spacing-md);
    text-align: center;
    border-bottom: 2px solid var(--primary-color);
    color: var(--white);
    font-weight: 600;
    font-size: var(--font-size-sm);
    position: sticky;
    top: 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.comparison-table td {
    padding: var(--spacing-md);
    border: 1px solid var(--gray-800);
    font-size: var(--font-size-sm);
    vertical-align: middle;
}

.comparison-table tbody tr:hover {
    background: var(--gray-900);
}

/* Column Styling with Enhanced Borders */
.ssor-cat, .cofdm-cat,
.ssoar-cat {
    font-weight: bold !important;
    background: rgba(26, 63, 136, 0.3) !important;
    border-left: 3px solid var(--primary-light) !important;
}

.ssor-val, .cofdm-val,
.ssoar-val {
    background: rgba(10, 100, 216, 0.3) !important;
    border-left: 3px solid var(--accent-color) !important;
}

.ssor-id,
.ssoar-id {
    background: rgba(219, 137, 52, 0.3) !important;
    border-left: 3px solid var(--warning-color) !important;
}

.cofdm-id {
    background: rgba(96, 125, 139, 0.3) !important;
    border-left: 3px solid var(--gray-500) !important;
}

.th-header {
    font-weight: 700 !important;
    font-size: var(--font-size-base) !important;
    letter-spacing: 0.1em !important;
}

/* Status Badges - Much more readable and prominent */
.status-badge {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-base);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--white);
    box-shadow: var(--shadow-sm);
    border: 1px solid transparent;
    display: inline-block;
    text-align: center;
}

.status-badge.status-match {
    background: var(--success-color);
    border-color: var(--white);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.status-badge.status-mismatch {
    background: var(--error-color);
    border-color: var(--white);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.status-badge.status-missing {
    background: var(--warning-color);
    color: var(--white);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

/* Enhanced Visibility for other status badges */
.status-badge.partial {
    background: var(--accent-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--success-color);
}

.status-badge.not-found {
    background: var(--error-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--error-color);
}

.status-badge.error {
    background: var(--warning-color);
    color: var(--white);
    border: 2px solid var(--white);
    text-shadow: 0 0 8px var(--warning-color);
}

/* Animations */
@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.7;
        transform: scale(1.1);
    }
}


/* --- Responsive Design --- */
@media (max-width: 768px) {
    .container {
        padding: 0 var(--spacing-md);
    }
    
    .header-content {
        flex-direction: column;
        gap: var(--spacing-md);
    }

    .form-grid {
        grid-template-columns: 1fr;
    }

    .form-row-split {
        flex-direction: column;
    }

    .scenario-grid {
        grid-template-columns: 1fr;
    }

    .summary-cards {
        grid-template-columns: repeat(2, 1fr);
    }

    .results-header {
        flex-direction: column;
        gap: var(--spacing-md);
        align-items: stretch;
    }
    
    .results-actions {
        justify-content: center;
        flex-wrap: wrap;
    }

    .tab-container {
        flex-direction: column;
    }

    .tab-button {
        width: 100%;
        justify-content: center;
    }

    .comparison-table {
        font-size: var(--font-size-sm);
    }
    .comparison-table th,
    .comparison-table td {
        padding: var(--spacing-sm);
    }
}

@media (max-width: 480px) {
    h1 {
        font-size: var(--font-size-xl);
    }
    .search-card {
        padding: var(--spacing-lg);
    }
    .summary-cards {
        grid-template-columns: 1fr;
    }
}

/* --- Utilities --- */
.hidden {
    display: none !important;
}

.text-center {
    text-align: center;
}

.text-success {
    color: var(--success-color);
}

.text-error {
    color: var(--error-color);
}

.text-warning {
    color: var(--warning-color);
}

/* Enhanced Focus States */
a:focus, button:focus {
    outline: 2px solid var(--accent-color);
    outline-offset: 2px;
}

/* --- Scrollbar Styling --- */
::-webkit-scrollbar {
    height: 8px;
    width: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-darker);
    border-radius: var(--radius-sm);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: var(--radius-sm);
    border: 2px solid var(--accent-color);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-light);
}


















<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Recon Portal</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/normalize/8.0.1/normalize.min.css">
    <link rel="stylesheet" href="/static/css/style.css">
    <link rel="stylesheet" href="/static/css/animation.css">
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Roboto+Mono:wght@400;700&family=Roboto:wght@400;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <header class="page-header">
        <div class="header-content">
            <div class="logo">
                <h1>Recon Portal</h1>
            </div>
            <nav class="main-nav"></nav>
            <div class="connection-status-container">
                <div id="connection-status" class="connection-status">
                    <span class="connection-dot"></span>
                    <span id="connection-text">Connecting...</span>
                </div>
            </div>
        </div>
    </header>

    <main class="main-content">
        <section class="intro-section fade-in">
            <div class="intro-content">
                <h1>Financial Data Reconciliation</h1>
                <p>Compare and synchronize financial trading data across systems.</p>
            </div>
        </section>

        <!-- Environment Selection Section -->
        <section class="environment-section">
            <div class="environment-card">
                <h2>Environment Selection</h2>
                <div class="form-group">
                    <label class="form-label" for="environment-select">Select Environment:</label>
                    <select id="environment-select" class="form-input">
                        <option value="dev">Development (DEV)</option>
                        <option value="qa" disabled>Quality Assurance (QA) - Coming Soon</option>
                        <option value="uat" disabled>User Acceptance Testing (UAT) - Coming Soon</option>
                        <option value="prod" disabled>Production (PROD) - Coming Soon</option>
                    </select>
                    <small class="form-hint">Select the target environment for reconciliation</small>
                </div>
            </div>
        </section>

        <!-- Reconciliation Type Selection -->
        <section class="recon-type-section">
            <div class="recon-type-card">
                <h2>Reconciliation Type</h2>
                <div class="tab-container">
                    <button class="tab-button active" data-tab="oscar-copper-star">
                        <i class="fas fa-database"></i> OSCAR - Copper - Star
                    </button>
                    <button class="tab-button" data-tab="oscar-copper-edb">
                        <i class="fas fa-server"></i> OSCAR - Copper - EDB
                    </button>
                </div>
            </div>
        </section>

        <!-- Reconciliation Form Section -->
        <section class="reconciliation-section" id="reconciliation-section">
            <div class="recon-header">
                <h2>Initiate Reconciliation</h2>
                <p>Enter GUID or combination of GFID and GUS ID</p>
            </div>

            <form id="reconcile-form">
                <div class="form-row form-row-centered">
                    <div class="form-group">
                        <label class="form-label" for="recon-id">GFID (GUID):</label>
                        <input type="text" id="recon-id" name="recon-id" class="form-input" placeholder="e.g., GFID251214">
                        <small class="form-hint">Global Firm ID (GUID) - enter alone</small>
                    </div>
                </div>
                <div class="form-row form-row-split">
                    <div class="form-group form-group-half">
                        <label class="form-label" for="fn-firm-id">GUB ID (GFID):</label>
                        <input type="text" id="fn-firm-id" name="fn-firm-id" class="form-input" placeholder="e.g., GUB_ID_FIRM_1">
                        <small class="form-hint">Enter with DFID below</small>
                    </div>
                    <div class="form-group form-group-half">
                        <label class="form-label" for="fn-user-id">DFID (GUS_ID):</label>
                        <input type="text" id="fn-user-id" name="fn-user-id" class="form-input" placeholder="e.g., DFID_USER_1">
                        <small class="form-hint">Enter with GUB ID above</small>
                    </div>
                </div>
                <div class="submit-container">
                    <button type="submit" class="btn btn-primary" id="submit-btn">
                        <span class="btn-text">Reconcile</span>
                    </button>
                </div>
            </form>
        </section>

        <!-- Results Section -->
        <section class="result-section" id="result-section">
            <div class="results-header">
                <h2>Reconciliation Results</h2>
                <button id="export-xml-btn" class="btn-secondary" style="display: none;">
                    <i class="fas fa-download"></i> Export XML
                </button>
            </div>
            
            <div class="single-comparison-table">
                <div class="table-header">
                    <h3>SSOAR vs COFDM Comparison</h3>
                </div>
                <div class="table-container">
                    <table id="comparison-table" class="comparison-table">
                        <thead class="table-header-sticky">
                            <tr>
                                <th scope="col" class="th-category">SSOAR-cat</th>
                                <th scope="col" class="th-value">SSOAR-val</th>
                                <th scope="col" class="th-id">SSOAR-id</th>
                                <th scope="col" class="th-status">Recon-Status</th>
                                <th scope="col" class="th-status">P-Final Status</th>
                                <th scope="col" class="th-id">COFDM-id</th>
                                <th scope="col" class="th-value">COFDM-val</th>
                                <th scope="col" class="th-category">COFDM-cat</th>
                            </tr>
                        </thead>
                        <tbody id="comparison-table-body">
                        </tbody>
                    </table>
                </div>
            </div>

            <div id="scenario-info-container" class="scenario-hidden">
            </div>
        </section>
    </main>
    
    <script src="/static/js/app.js"></script>
</body>
</html>
