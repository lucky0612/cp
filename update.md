<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool - Lookup</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="style.css">
</head>
<body>
    <!-- Loading Overlay -->
    <div id="loading-overlay" class="loading-overlay hidden">
        <div class="loading-spinner">
            <div class="spinner"></div>
            <p>Looking up data...</p>
        </div>
    </div>

    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-search"></i>
                    <h1>OSCAR Data Lookup</h1>
                </div>
                <div class="header-actions">
                    <button class="btn-icon" id="health-check-btn" title="Check System Health">
                        <i class="fas fa-heartbeat"></i>
                    </button>
                    <div class="connection-status" id="connection-status">
                        <div class="status-dot connected"></div>
                        <span>Connected</span>
                    </div>
                </div>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="main-content">
        <div class="container">
            <!-- Welcome Section -->
            <section class="welcome-section fade-in">
                <div class="welcome-content">
                    <h2>OSCAR Trading Data Lookup</h2>
                    <p>Enter any field to automatically populate related trading data across all systems</p>
                </div>
            </section>

            <!-- Lookup Form -->
            <section class="lookup-section">
                <div class="lookup-card slide-up">
                    <div class="lookup-header">
                        <h3><i class="fas fa-database"></i> Data Lookup & Auto-Fill</h3>
                        <p>Enter any field below and the system will automatically fill related information</p>
                    </div>
                    
                    <form id="lookup-form">
                        <!-- Date Selection -->
                        <div class="form-row">
                            <div class="form-group full-width">
                                <label for="lookup-date">
                                    <i class="fas fa-calendar"></i> Lookup Date
                                </label>
                                <input type="date" id="lookup-date" name="date" required>
                                <div class="input-info">Select the date for data lookup (defaults to today)</div>
                            </div>
                        </div>

                        <!-- Input Fields Grid -->
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="guid">
                                    <i class="fas fa-key"></i> GUID
                                    <span class="field-length">(12 chars)</span>
                                </label>
                                <div class="input-wrapper">
                                    <input 
                                        type="text" 
                                        id="guid" 
                                        name="guid" 
                                        placeholder="e.g., JUY3TALM2SSP" 
                                        maxlength="12"
                                        class="auto-lookup-field"
                                        data-field-type="guid"
                                    >
                                    <div class="input-status" id="guid-status"></div>
                                </div>
                                <div class="input-info">Global Unique Identifier</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gus-id">
                                    <i class="fas fa-user"></i> GUS ID
                                    <span class="field-length">(5 chars)</span>
                                </label>
                                <div class="input-wrapper">
                                    <input 
                                        type="text" 
                                        id="gus-id" 
                                        name="gus_id" 
                                        placeholder="e.g., ABCDE" 
                                        maxlength="5"
                                        class="auto-lookup-field"
                                        data-field-type="gus_id"
                                    >
                                    <div class="input-status" id="gus-id-status"></div>
                                </div>
                                <div class="input-info">Globex User Signature ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gfid">
                                    <i class="fas fa-building"></i> GFID
                                    <span class="field-length">(4 chars)</span>
                                </label>
                                <div class="input-wrapper">
                                    <input 
                                        type="text" 
                                        id="gfid" 
                                        name="gfid" 
                                        placeholder="e.g., ABCD" 
                                        maxlength="4"
                                        class="auto-lookup-field"
                                        data-field-type="gfid"
                                    >
                                    <div class="input-status" id="gfid-status"></div>
                                </div>
                                <div class="input-info">Globex Firm ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="session-id">
                                    <i class="fas fa-plug"></i> Session ID
                                </label>
                                <div class="input-wrapper">
                                    <input 
                                        type="text" 
                                        id="session-id" 
                                        name="session_id" 
                                        placeholder="e.g., MDBLZ, FIF" 
                                        class="auto-lookup-field"
                                        data-field-type="session_id"
                                    >
                                    <div class="input-status" id="session-id-status"></div>
                                </div>
                                <div class="input-info">Trading session identifier</div>
                            </div>
                        </div>

                        <!-- Action Buttons -->
                        <div class="form-actions">
                            <button type="button" class="btn-secondary" id="clear-btn">
                                <i class="fas fa-times"></i> Clear All
                            </button>
                            <button type="submit" class="btn-primary" id="lookup-btn">
                                <i class="fas fa-search"></i> Manual Lookup
                            </button>
                        </div>
                    </form>
                </div>
            </section>

            <!-- Database Verification Section -->
            <section class="verification-section" id="verification-section" style="display: none;">
                <div class="verification-header">
                    <h3><i class="fas fa-shield-alt"></i> Database Verification</h3>
                    <p>Check which databases contain your data</p>
                </div>

                <div class="db-verification-grid">
                    <div class="db-card oscar" id="oscar-verification">
                        <div class="db-header">
                            <div class="db-icon">
                                <i class="fas fa-database"></i>
                            </div>
                            <div class="db-info">
                                <h4>OSCAR</h4>
                                <p>Primary Trading System</p>
                            </div>
                        </div>
                        <div class="db-status" id="oscar-status">
                            <div class="status-indicator checking">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <span class="status-text">Checking...</span>
                        </div>
                        <div class="db-details" id="oscar-details">
                            <div class="detail-item">
                                <span class="label">Records Found:</span>
                                <span class="value" id="oscar-count">-</span>
                            </div>
                            <div class="detail-item">
                                <span class="label">Last Updated:</span>
                                <span class="value" id="oscar-updated">-</span>
                            </div>
                        </div>
                    </div>

                    <div class="db-card copper" id="copper-verification">
                        <div class="db-header">
                            <div class="db-icon">
                                <i class="fas fa-database"></i>
                            </div>
                            <div class="db-info">
                                <h4>CoPPER</h4>
                                <p>Settlement System</p>
                            </div>
                        </div>
                        <div class="db-status" id="copper-status">
                            <div class="status-indicator checking">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <span class="status-text">Checking...</span>
                        </div>
                        <div class="db-details" id="copper-details">
                            <div class="detail-item">
                                <span class="label">Records Found:</span>
                                <span class="value" id="copper-count">-</span>
                            </div>
                            <div class="detail-item">
                                <span class="label">Last Updated:</span>
                                <span class="value" id="copper-updated">-</span>
                            </div>
                        </div>
                    </div>

                    <div class="db-card star" id="star-verification">
                        <div class="db-header">
                            <div class="db-icon">
                                <i class="fas fa-database"></i>
                            </div>
                            <div class="db-info">
                                <h4>STAR</h4>
                                <p>Risk Management</p>
                            </div>
                        </div>
                        <div class="db-status" id="star-status">
                            <div class="status-indicator checking">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <span class="status-text">Checking...</span>
                        </div>
                        <div class="db-details" id="star-details">
                            <div class="detail-item">
                                <span class="label">Records Found:</span>
                                <span class="value" id="star-count">-</span>
                            </div>
                            <div class="detail-item">
                                <span class="label">Last Updated:</span>
                                <span class="value" id="star-updated">-</span>
                            </div>
                        </div>
                    </div>

                    <div class="db-card edb" id="edb-verification">
                        <div class="db-header">
                            <div class="db-icon">
                                <i class="fas fa-database"></i>
                            </div>
                            <div class="db-info">
                                <h4>EDB</h4>
                                <p>Entity Database</p>
                            </div>
                        </div>
                        <div class="db-status" id="edb-status">
                            <div class="status-indicator checking">
                                <i class="fas fa-spinner fa-spin"></i>
                            </div>
                            <span class="status-text">Checking...</span>
                        </div>
                        <div class="db-details" id="edb-details">
                            <div class="detail-item">
                                <span class="label">Records Found:</span>
                                <span class="value" id="edb-count">-</span>
                            </div>
                            <div class="detail-item">
                                <span class="label">Last Updated:</span>
                                <span class="value" id="edb-updated">-</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Summary Section -->
                <div class="verification-summary" id="verification-summary" style="display: none;">
                    <div class="summary-card">
                        <div class="summary-header">
                            <h4><i class="fas fa-chart-pie"></i> Verification Summary</h4>
                        </div>
                        <div class="summary-stats">
                            <div class="stat-item">
                                <span class="stat-value success" id="found-count">0</span>
                                <span class="stat-label">Databases Found</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value error" id="missing-count">0</span>
                                <span class="stat-label">Missing From</span>
                            </div>
                            <div class="stat-item">
                                <span class="stat-value" id="total-records">0</span>
                                <span class="stat-label">Total Records</span>
                            </div>
                        </div>
                        <div class="summary-actions">
                            <button class="btn-primary" id="proceed-btn" disabled>
                                <i class="fas fa-arrow-right"></i> Proceed to Reconciliation
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Results Section (Auto-filled data) -->
            <section class="results-section" id="results-section" style="display: none;">
                <div class="results-header">
                    <h3><i class="fas fa-magic"></i> Auto-Filled Results</h3>
                    <div class="results-actions">
                        <button class="btn-secondary" id="copy-results-btn">
                            <i class="fas fa-copy"></i> Copy Data
                        </button>
                        <button class="btn-secondary" id="edit-results-btn">
                            <i class="fas fa-edit"></i> Edit Values
                        </button>
                    </div>
                </div>

                <div class="results-grid" id="results-grid">
                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-key"></i> GUID</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-guid">-</div>
                            <div class="card-source" id="result-guid-source">Auto-filled</div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-user"></i> GUS ID</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-gus-id">-</div>
                            <div class="card-source" id="result-gus-id-source">Auto-filled</div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-building"></i> GFID</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-gfid">-</div>
                            <div class="card-source" id="result-gfid-source">Auto-filled</div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-plug"></i> Session ID</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-session-id">-</div>
                            <div class="card-source" id="result-session-id-source">Auto-filled</div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-exchange-alt"></i> Exchange</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-exchange">-</div>
                            <div class="card-source" id="result-exchange-source">From OSCAR</div>
                        </div>
                    </div>

                    <div class="result-card">
                        <div class="card-header">
                            <h4><i class="fas fa-info-circle"></i> Status</h4>
                        </div>
                        <div class="card-content">
                            <div class="card-value" id="result-status">-</div>
                            <div class="card-source" id="result-status-source">Current</div>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <!-- Toast Container -->
    <div class="toast-container" id="toast-container"></div>

    <!-- Footer -->
    <footer class="footer">
        <div class="container">
            <p>&copy; 2024 OSCAR Reconciliation Tool. Auto-fill functionality powered by real-time database lookups.</p>
        </div>
    </footer>

    <script src="script.js"></script>
</body>
</html>











/* OSCAR Reconciliation Tool - Lookup Page Styles */

/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Color Palette */
    --primary-color: #1a365d;
    --primary-light: #2c5aa0;
    --primary-dark: #0f2537;
    --secondary-color: #e53e3e;
    --accent-color: #00b4d8;
    --success-color: #38a169;
    --warning-color: #ed8936;
    --error-color: #e53e3e;
    
    /* Neutral Colors */
    --white: #ffffff;
    --gray-50: #f9fafb;
    --gray-100: #f3f4f6;
    --gray-200: #e5e7eb;
    --gray-300: #d1d5db;
    --gray-400: #9ca3af;
    --gray-500: #6b7280;
    --gray-600: #4b5563;
    --gray-700: #374151;
    --gray-800: #1f2937;
    --gray-900: #111827;
    
    /* System Colors */
    --oscar-color: #1a365d;
    --copper-color: #00b4d8;
    --star-color: #38a169;
    --edb-color: #ed8936;
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #48bb78 100%);
    --gradient-warning: linear-gradient(135deg, var(--warning-color) 0%, #f6ad55 100%);
    --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #fc8181 100%);
    --gradient-oscar: linear-gradient(135deg, var(--oscar-color) 0%, #2c5aa0 100%);
    --gradient-copper: linear-gradient(135deg, var(--copper-color) 0%, #0ea5e9 100%);
    --gradient-star: linear-gradient(135deg, var(--star-color) 0%, #48bb78 100%);
    --gradient-edb: linear-gradient(135deg, var(--edb-color) 0%, #f6ad55 100%);
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    
    /* Spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;
    --spacing-3xl: 4rem;
    
    /* Border Radius */
    --radius-sm: 0.375rem;
    --radius-md: 0.5rem;
    --radius-lg: 0.75rem;
    --radius-xl: 1rem;
    
    /* Shadows */
    --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    
    /* Transitions */
    --transition-fast: 0.15s ease-in-out;
    --transition-normal: 0.3s ease-in-out;
    --transition-slow: 0.5s ease-in-out;
}

/* Base Typography */
body {
    font-family: var(--font-family);
    font-size: 1rem;
    line-height: 1.6;
    color: var(--gray-800);
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    min-height: 100vh;
}

/* Layout */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

/* Header */
.header {
    background: var(--gradient-primary);
    color: var(--white);
    padding: var(--spacing-lg) 0;
    box-shadow: var(--shadow-lg);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
}

.logo i {
    font-size: 1.5rem;
    color: var(--accent-color);
}

.logo h1 {
    font-size: 1.5rem;
    font-weight: 700;
    margin: 0;
}

.header-actions {
    display: flex;
    align-items: center;
    gap: var(--spacing-lg);
}

.btn-icon {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    color: var(--white);
    padding: var(--spacing-sm);
    border-radius: var(--radius-md);
    cursor: pointer;
    transition: var(--transition-normal);
    backdrop-filter: blur(10px);
}

.btn-icon:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-2px);
}

.connection-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    background: rgba(255, 255, 255, 0.1);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(10px);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--warning-color);
    animation: pulse 2s infinite;
}

.status-dot.connected {
    background: var(--success-color);
}

.status-dot.disconnected {
    background: var(--error-color);
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* Main Content */
.main-content {
    padding: var(--spacing-2xl) 0;
    min-height: calc(100vh - 200px);
}

/* Welcome Section */
.welcome-section {
    text-align: center;
    margin-bottom: var(--spacing-2xl);
}

.welcome-content h2 {
    font-size: 1.875rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: var(--spacing-md);
}

.welcome-content p {
    font-size: 1.125rem;
    color: var(--gray-600);
    max-width: 600px;
    margin: 0 auto;
}

/* Lookup Section */
.lookup-section {
    margin-bottom: var(--spacing-2xl);
}

.lookup-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-2xl);
    box-shadow: var(--shadow-xl);
    border: 1px solid var(--gray-200);
    max-width: 900px;
    margin: 0 auto;
}

.lookup-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
}

.lookup-header h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: var(--spacing-sm);
}

.lookup-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.lookup-header p {
    color: var(--gray-600);
    font-size: 0.875rem;
}

/* Form Styles */
.form-row {
    margin-bottom: var(--spacing-lg);
}

.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.form-group.full-width {
    max-width: 400px;
    margin: 0 auto;
}

.form-group label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--gray-700);
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
}

.form-group label i {
    color: var(--accent-color);
}

.field-length {
    font-size: 0.75rem;
    color: var(--gray-500);
    font-weight: 400;
}

.input-wrapper {
    position: relative;
}

.form-group input {
    width: 100%;
    padding: var(--spacing-md);
    border: 2px solid var(--gray-300);
    border-radius: var(--radius-lg);
    font-size: 1rem;
    transition: var(--transition-normal);
    background: var(--white);
    font-family: var(--font-family);
    text-transform: uppercase;
    padding-right: 3rem;
}

.form-group input:focus {
    outline: none;
    border-color: var(--primary-color);
    box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
}

.form-group input.valid {
    border-color: var(--success-color);
    box-shadow: 0 0 0 3px rgba(56, 161, 105, 0.1);
}

.form-group input.invalid {
    border-color: var(--error-color);
    box-shadow: 0 0 0 3px rgba(229, 62, 62, 0.1);
}

.form-group input.auto-filled {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(0, 180, 216, 0.1);
    font-weight: 600;
}

.input-status {
    position: absolute;
    right: var(--spacing-md);
    top: 50%;
    transform: translateY(-50%);
    font-size: 0.875rem;
}

.input-status.loading {
    color: var(--warning-color);
}

.input-status.success {
    color: var(--success-color);
}

.input-status.error {
    color: var(--error-color);
}

.input-info {
    font-size: 0.75rem;
    color: var(--gray-500);
}

/* Buttons */
.btn-primary {
    background: var(--gradient-primary);
    color: var(--white);
    border: none;
    padding: var(--spacing-md) var(--spacing-xl);
    border-radius: var(--radius-lg);
    font-size: 1rem;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
    box-shadow: var(--shadow-md);
    white-space: nowrap;
}

.btn-primary:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.btn-secondary {
    background: var(--white);
    color: var(--gray-700);
    border: 2px solid var(--gray-300);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.btn-secondary:hover {
    border-color: var(--primary-color);
    color: var(--primary-color);
    transform: translateY(-1px);
}

.form-actions {
    display: flex;
    justify-content: center;
    gap: var(--spacing-md);
    margin-top: var(--spacing-xl);
}

/* Database Verification Section */
.verification-section {
    margin-bottom: var(--spacing-2xl);
}

.verification-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
}

.verification-header h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--primary-color);
    margin-bottom: var(--spacing-sm);
}

.verification-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.verification-header p {
    color: var(--gray-600);
    font-size: 0.875rem;
}

.db-verification-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.db-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--gray-200);
    transition: var(--transition-normal);
    position: relative;
    overflow: hidden;
}

.db-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 4px;
    background: var(--gray-300);
    transition: var(--transition-normal);
}

.db-card.oscar::before {
    background: var(--gradient-oscar);
}

.db-card.copper::before {
    background: var(--gradient-copper);
}

.db-card.star::before {
    background: var(--gradient-star);
}

.db-card.edb::before {
    background: var(--gradient-edb);
}

.db-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
}

.db-card.found {
    border-color: var(--success-color);
    background: linear-gradient(135deg, rgba(56, 161, 105, 0.05) 0%, rgba(72, 187, 120, 0.05) 100%);
}

.db-card.missing {
    border-color: var(--error-color);
    background: linear-gradient(135deg, rgba(229, 62, 62, 0.05) 0%, rgba(252, 129, 129, 0.05) 100%);
}

.db-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
}

.db-icon {
    width: 48px;
    height: 48px;
    border-radius: var(--radius-lg);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.25rem;
    color: var(--white);
}

.oscar .db-icon {
    background: var(--gradient-oscar);
}

.copper .db-icon {
    background: var(--gradient-copper);
}

.star .db-icon {
    background: var(--gradient-star);
}

.edb .db-icon {
    background: var(--gradient-edb);
}

.db-info h4 {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--gray-800);
    margin-bottom: var(--spacing-xs);
}

.db-info p {
    font-size: 0.875rem;
    color: var(--gray-500);
    margin: 0;
}

.db-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-md);
    margin-bottom: var(--spacing-lg);
    padding: var(--spacing-md);
    border-radius: var(--radius-lg);
    background: var(--gray-50);
}

.status-indicator {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
}

.status-indicator.checking {
    background: var(--warning-color);
    color: var(--white);
}

.status-indicator.found {
    background: var(--success-color);
    color: var(--white);
}

.status-indicator.missing {
    background: var(--error-color);
    color: var(--white);
}

.status-text {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--gray-700);
}

.db-details {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.detail-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-sm);
    background: var(--gray-50);
    border-radius: var(--radius-md);
}

.detail-item .label {
    font-size: 0.75rem;
    color: var(--gray-600);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.detail-item .value {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--gray-800);
}

/* Verification Summary */
.verification-summary {
    max-width: 600px;
    margin: 0 auto;
}

.summary-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    box-shadow: var(--shadow-xl);
    border: 1px solid var(--gray-200);
    text-align: center;
}

.summary-header {
    margin-bottom: var(--spacing-lg);
}

.summary-header h4 {
    font-size: 1.125rem;
    font-weight: 600;
    color: var(--primary-color);
}

.summary-header h4 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.summary-stats {
    display: flex;
    justify-content: space-around;
    margin-bottom: var(--spacing-xl);
}

.stat-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: var(--spacing-xs);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
}

.stat-value.success {
    color: var(--success-color);
}

.stat-value.error {
    color: var(--error-color);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--gray-600);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.summary-actions {
    display: flex;
    justify-content: center;
}

/* Results Section */
.results-section {
    margin-bottom: var(--spacing-2xl);
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xl);
}

.results-header h3 {
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--primary-color);
}

.results-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.results-actions {
    display: flex;
    gap: var(--spacing-md);
}

.results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.result-card {
    background: var(--white);
    border-radius: var(--radius-xl);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-lg);
    border: 1px solid var(--gray-200);
    transition: var(--transition-normal);
}

.result-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
}

.card-header {
    display: flex;
    align-items: center;
    margin-bottom: var(--spacing-md);
}

.card-header h4 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--gray-800);
    margin: 0;
}

.card-header h4 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.card-content {
    text-align: center;
}

.card-value {
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: var(--spacing-xs);
    font-family: 'Courier New', monospace;
    word-break: break-all;
    text-transform: uppercase;
}

.card-source {
    font-size: 0.75rem;
    color: var(--gray-500);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Loading Overlay */
.loading-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.95);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 2000;
    backdrop-filter: blur(10px);
}

.loading-spinner {
    text-align: center;
}

.spinner {
    width: 50px;
    height: 50px;
    border: 4px solid var(--gray-300);
    border-top: 4px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto var(--spacing-lg);
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.loading-spinner p {
    color: var(--gray-700);
    font-weight: 500;
}

/* Toast Notifications */
.toast-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 3000;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
}

.toast {
    background: var(--white);
    padding: var(--spacing-lg);
    border-radius: var(--radius-lg);
    box-shadow: var(--shadow-xl);
    border-left: 4px solid;
    max-width: 400px;
    animation: slideInRight 0.3s ease-out;
}

.toast.success {
    border-left-color: var(--success-color);
}

.toast.error {
    border-left-color: var(--error-color);
}

.toast.warning {
    border-left-color: var(--warning-color);
}

.toast.info {
    border-left-color: var(--accent-color);
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
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

@keyframes slideInRight {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes bounceIn {
    0% {
        opacity: 0;
        transform: scale(0.3);
    }
    50% {
        opacity: 1;
        transform: scale(1.05);
    }
    70% {
        transform: scale(0.9);
    }
    100% {
        opacity: 1;
        transform: scale(1);
    }
}

.fade-in {
    animation: fadeIn 0.8s ease-out;
}

.slide-up {
    animation: fadeInUp 0.8s ease-out;
}

.bounce-in {
    animation: bounceIn 0.6s ease-out;
}

/* Utilities */
.hidden {
    display: none !important;
}

/* Footer */
.footer {
    background: var(--gray-800);
    color: var(--white);
    text-align: center;
    padding: var(--spacing-xl) 0;
    margin-top: auto;
}

/* Responsive Design */
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
    
    .db-verification-grid {
        grid-template-columns: 1fr;
    }
    
    .results-header {
        flex-direction: column;
        gap: var(--spacing-md);
        align-items: stretch;
    }
    
    .results-actions {
        justify-content: center;
    }
    
    .form-actions {
        flex-direction: column;
    }
    
    .results-grid {
        grid-template-columns: 1fr;
    }
    
    .summary-stats {
        flex-direction: column;
        gap: var(--spacing-lg);
    }
}

@media (max-width: 480px) {
    .welcome-content h2 {
        font-size: 1.5rem;
    }
    
    .lookup-card {
        padding: var(--spacing-lg);
    }
    
    .db-card {
        padding: var(--spacing-md);
    }
}
