<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" href="{{ url_for('static', filename='css/animation.css') }}">
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-exchange-alt"></i>
                    <h1>OSCAR Reconciliation Tool</h1>
                </div>
                <div class="connection-status">
                    <span class="status-dot connected" id="connection-status"></span>
                    <span id="connection-text">Connected</span>
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
                    <h2>Data Reconciliation Between OSCAR, CoPPER, STAR & EDB</h2>
                    <p>Compare and synchronize financial trading data across multiple systems</p>
                </div>
            </section>

            <!-- Search Section -->
            <section class="search-section slide-up">
                <div class="search-card">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Configure your reconciliation criteria and comparison settings</p>
                    </div>

                    <form id="reconcile-form">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="reconcile-date"><i class="fas fa-calendar"></i> Reconciliation Date</label>
                                <input type="date" id="reconcile-date" name="reconcile-date" required>
                                <div class="input-info">Select the date for data comparison</div>
                            </div>
                            <div class="form-group">
                                <label for="guid"><i class="fas fa-key"></i> GUID</label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., ABCDEFGH1234">
                                <div class="input-info">Global Unique Identifier (any length)</div>
                            </div>
                            <div class="form-group">
                                <label for="gfid"><i class="fas fa-building"></i> GFID</label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., BTEC">
                                <div class="input-info">Globex Firm ID (any length)</div>
                            </div>
                            <div class="form-group">
                                <label for="gus-id"><i class="fas fa-user"></i> GUS ID</label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., GUS01">
                                <div class="input-info">Globex User Signature ID (any length)</div>
                            </div>
                            <div class="form-group">
                                <label for="contact-id"><i class="fas fa-address-book"></i> Contact ID</label>
                                <input type="text" id="contact-id" name="contact_id" placeholder="Contact Identifier">
                                <div class="input-info">Associated Contact Identifier</div>
                            </div>
                            <div class="form-group">
                                <label for="session-id"><i class="fas fa-plug"></i> Session ID</label>
                                <input type="text" id="session-id" name="session_id" placeholder="e.g., MDBLZ, FIF">
                                <div class="input-info">Trading Session Identifier</div>
                            </div>
                        </div>
                        
                        <div class="scenario-selector">
                            <h4><i class="fas fa-cogs"></i> Comparison Scenarios</h4>
                            <div class="scenario-grid">
                                <div class="form-group">
                                    <label for="comparison-type">Primary Comparison</label>
                                    <select id="comparison-type" name="comparison_type">
                                        <option value="guid_lookup">Standard GUID Lookup</option>
                                        <option value="scenario_2_1">Scenario 2.1 - Both Expired</option>
                                        <option value="scenario_2_2">Scenario 2.2 - OSCAR Expired, COPPER Active</option>
                                        <option value="scenario_2_3">Scenario 2.3 - OSCAR Expired, COPPER Missing</option>
                                        <option value="scenario_2_4">Scenario 2.4 - OSCAR Active, COPPER Missing</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="comparison-field">Compare By</label>
                                    <select id="comparison-field" name="comparison_field">
                                        <option value="session_id">Session ID</option>
                                        <option value="gus_id">GUS ID</option>
                                        <option value="gfid">GFID</option>
                                        <option value="contact_id">Contact ID</option>
                                        <option value="product_id">Product ID</option>
                                        <option value="status">Status</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="table-set">Table Set</label>
                                    <select id="table-set" name="table_set">
                                        <option value="both_star_edb">Both STAR & EDB</option>
                                        <option value="star_only">STAR Only</option>
                                        <option value="edb_only">EDB Only</option>
                                    </select>
                                </div>
                                <div class="form-group">
                                    <label for="sync-direction">Sync Direction</label>
                                    <select id="sync-direction" name="sync_direction">
                                        <option value="bidirectional">Bidirectional</option>
                                        <option value="oscar_to_systems">OSCAR → Systems</option>
                                        <option value="systems_to_oscar">Systems → OSCAR</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="submit-container">
                            <button type="submit" class="btn-primary" id="submit-btn">
                                <i class="fas fa-sync-alt"></i>
                                <span>Execute Reconciliation</span>
                            </button>
                        </div>
                    </form>
                </div>
            </section>
            
            <section class="results-section" id="results-section">
                <div class="results-header">
                    <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                    <div class="results-actions">
                        <button class="btn-secondary" id="export-btn">
                            <i class="fas fa-file-download"></i> Export Results
                        </button>
                        <button class="btn-secondary" id="refresh-btn">
                            <i class="fas fa-redo"></i> Refresh Data
                        </button>
                        <button class="btn-secondary" id="clear-results-btn">
                            <i class="fas fa-times"></i> Clear Results
                        </button>
                    </div>
                </div>

                <div class="summary-cards">
                    <div class="summary-card total">
                        <div class="card-number" id="total-records">259</div>
                        <div class="card-label">Total Records</div>
                    </div>
                    <div class="summary-card matches">
                        <div class="card-number" id="total-matches">193</div>
                        <div class="card-label">Matches</div>
                    </div>
                    <div class="summary-card mismatches">
                        <div class="card-number" id="total-mismatches">54</div>
                        <div class="card-label">Mismatches</div>
                    </div>
                    <div class="summary-card missing">
                        <div class="card-number" id="total-missing">12</div>
                        <div class="card-label">Missing Records</div>
                    </div>
                </div>

                <div class="comparison-tables">
                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ STAR Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table">
                                <thead>
                                    <tr>
                                        <th rowspan="2">Record ID</th>
                                        <th colspan="3" class="db-header oscar-col">OSCAR</th>
                                        <th colspan="3" class="db-header copper-col">CoPPER</th>
                                        <th colspan="3" class="db-header star-col">STAR</th>
                                        <th rowspan="2">Status</th>
                                        <th rowspan="2">Actions</th>
                                    </tr>
                                    <tr>
                                        <th class="oscar-col">GUID</th>
                                        <th class="oscar-col">Status</th>
                                        <th class="oscar-col">Last Updated</th>
                                        <th class="copper-col">GFID</th>
                                        <th class="copper-col">GUS ID</th>
                                        <th class="copper-col">Session ID</th>
                                        <th class="star-col">Product ID</th>
                                        <th class="star-col">Status</th>
                                        <th class="star-col">Settlement</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>001</td>
                                        <td class="oscar-col">TESTGUID001</td>
                                        <td class="oscar-col">ACTIVE</td>
                                        <td class="oscar-col">2024-01-15</td>
                                        <td class="copper-col">TEST</td>
                                        <td class="copper-col">GUS01</td>
                                        <td class="copper-col">MDBLZ</td>
                                        <td class="star-col">ACTIVES</td>
                                        <td class="star-col">ACTIVE</td>
                                        <td class="star-col">COMPLETE</td>
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-eye"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>002</td>
                                        <td class="oscar-col">TESTGUID002</td>
                                        <td class="oscar-col">EXPIRED</td>
                                        <td class="oscar-col">2023-12-31</td>
                                        <td class="copper-col">TEST</td>
                                        <td class="copper-col">GUS02</td>
                                        <td class="copper-col">FIF</td>
                                        <td class="star-col">-</td>
                                        <td class="star-col">-</td>
                                        <td class="star-col">-</td>
                                        <td><span class="status-badge status-mismatch">MISMATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-sync"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>003</td>
                                        <td class="oscar-col">TESTGUID003</td>
                                        <td class="oscar-col"></td>
                                        <td class="oscar-col"></td>
                                        <td class="copper-col">BTEC</td>
                                        <td class="copper-col">GUS01</td>
                                        <td class="copper-col">FIF</td>
                                        <td class="star-col">ACTIVES</td>
                                        <td class="star-col">PENDING</td>
                                        <td class="star-col">PENDING</td>
                                        <td><span class="status-badge status-missing">MISSING</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-plus"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>004</td>
                                        <td class="oscar-col">TESTGUID004</td>
                                        <td class="oscar-col">ACTIVE</td>
                                        <td class="oscar-col">2024-01-12</td>
                                        <td class="copper-col">TEST</td>
                                        <td class="copper-col">GUS04</td>
                                        <td class="copper-col">MDBLZ</td>
                                        <td class="star-col">ACTIVES</td>
                                        <td class="star-col">ACTIVE</td>
                                        <td class="star-col">COMPLETE</td>
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-eye"></i></button></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ EDB Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table">
                                <thead>
                                    <tr>
                                        <th rowspan="2">Record ID</th>
                                        <th colspan="3" class="db-header oscar-col">OSCAR</th>
                                        <th colspan="3" class="db-header copper-col">CoPPER</th>
                                        <th colspan="3" class="db-header edb-col">EDB</th>
                                        <th rowspan="2">Status</th>
                                        <th rowspan="2">Actions</th>
                                    </tr>
                                    <tr>
                                        <th class="oscar-col">GUID</th>
                                        <th class="oscar-col">GUS ID</th>
                                        <th class="oscar-col">Contact ID</th>
                                        <th class="copper-col">Session ID</th>
                                        <th class="copper-col">Product</th>
                                        <th class="copper-col">Permission</th>
                                        <th class="edb-col">Entity ID</th>
                                        <th class="edb-col">Type</th>
                                        <th class="edb-col">Schema</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <tr>
                                        <td>E001</td>
                                        <td class="oscar-col">TESTGUID005</td>
                                        <td class="oscar-col">GUS01</td>
                                        <td class="oscar-col">CONT001</td>
                                        <td class="copper-col">MDBLZ</td>
                                        <td class="copper-col">BTEC_EU</td>
                                        <td class="copper-col">READ_WRITE</td>
                                        <td class="edb-col">ENT001</td>
                                        <td class="edb-col">USER</td>
                                        <td class="edb-col">TRADING</td>
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-eye"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>E002</td>
                                        <td class="oscar-col">TESTGUID006</td>
                                        <td class="oscar-col">GUS02</td>
                                        <td class="oscar-col">CONT002</td>
                                        <td class="copper-col">FIF</td>
                                        <td class="copper-col">EBS</td>
                                        <td class="copper-col">READ_ONLY</td>
                                        <td class="edb-col">-</td>
                                        <td class="edb-col">-</td>
                                        <td class="edb-col">-</td>
                                        <td><span class="status-badge status-missing">MISSING</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-plus"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>E003</td>
                                        <td class="oscar-col">TESTGUID007</td>
                                        <td class="oscar-col">GUS03</td>
                                        <td class="oscar-col">CONT003</td>
                                        <td class="copper-col">MDBLZ</td>
                                        <td class="copper-col">CME_FO</td>
                                        <td class="copper-col">ADMIN</td>
                                        <td class="edb-col">ENT003</td>
                                        <td class="edb-col">ADMIN</td>
                                        <td class="edb-col">MANAGEMENT</td>
                                        <td><span class="status-badge status-mismatch">MISMATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-sync"></i></button></td>
                                    </tr>
                                    <tr>
                                        <td>E004</td>
                                        <td class="oscar-col">TESTGUID008</td>
                                        <td class="oscar-col">GUS04</td>
                                        <td class="oscar-col">CONT004</td>
                                        <td class="copper-col">FIF</td>
                                        <td class="copper-col">BTEC_US</td>
                                        <td class="copper-col">READ_WRITE</td>
                                        <td class="edb-col">ENT004</td>
                                        <td class="edb-col">USER</td>
                                        <td class="edb-col">TRADING</td>
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td><button class="btn-secondary table-action"><i class="fas fa-eye"></i></button></td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script src="{{ url_for('static', filename='js/dynamic_javascript_original_ui.js') }}"></script>
</body>
</html>






















/* Reset and Base Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Color Palette - Using exact provided colors */
    --primary-color: #1a365d;
    --primary-light: #2c5aa0;
    --primary-dark: #0f2537;
    --secondary-color: #e53e3e;
    --accent-color: #00b4d8;
    --success-color: #38a169;
    --warning-color: #ed8936;
    --error-color: #e53e3e;
    
    /* Neutral Colors - Dark theme */
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
    
    /* Dark Background Variants */
    --bg-primary: #0f1419;
    --bg-secondary: #1a252f;
    --bg-card: #2d3748;
    --bg-darker: #1a202c;
    
    /* Gradients */
    --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
    --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #48bb78 100%);
    --gradient-warning: linear-gradient(135deg, var(--warning-color) 0%, #f6ad55 100%);
    --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #fc8181 100%);
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 1.875rem;
    --font-size-4xl: 2.25rem;
    
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
    
    /* Shadows - Enhanced for depth */
    --shadow-sm: 0 2px 4px 0 rgba(0, 0, 0, 0.1);
    --shadow-md: 0 6px 12px -2px rgba(0, 0, 0, 0.15), 0 4px 8px -2px rgba(0, 0, 0, 0.1);
    --shadow-lg: 0 12px 24px -4px rgba(0, 0, 0, 0.2), 0 8px 16px -4px rgba(0, 0, 0, 0.1);
    --shadow-xl: 0 24px 48px -8px rgba(0, 0, 0, 0.25), 0 16px 32px -8px rgba(0, 0, 0, 0.15);
    
    /* Transitions */
    --transition-fast: 0.15s ease-in-out;
    --transition-normal: 0.3s ease-in-out;
    --transition-slow: 0.5s ease-in-out;
}

/* Base Typography */
body {
    font-family: var(--font-family);
    font-size: var(--font-size-base);
    line-height: 1.6;
    color: var(--white);
    background: var(--bg-primary);
    min-height: 100vh;
}

/* Layout */
.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 0 var(--spacing-lg);
}

/* Header */
.header {
    background: var(--gradient-primary);
    color: var(--white);
    padding: var(--spacing-lg) 0;
    box-shadow: var(--shadow-xl);
    position: sticky;
    top: 0;
    z-index: 100;
    border-bottom: 2px solid var(--accent-color);
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
    font-size: var(--font-size-2xl);
    color: var(--accent-color);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.logo h1 {
    font-size: var(--font-size-2xl);
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.connection-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    background: rgba(255, 255, 255, 0.15);
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-lg);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: var(--shadow-md);
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--warning-color);
    animation: pulse 2s infinite;
    box-shadow: 0 0 8px currentColor;
}

.status-dot.connected {
    background: var(--success-color);
}

.status-dot.disconnected {
    background: var(--error-color);
}

/* Main Content */
.main-content {
    padding: var(--spacing-2xl) 0;
    min-height: calc(100vh - 200px);
}

/* Welcome Section */
.welcome-section {
    text-align: center;
    margin-bottom: var(--spacing-3xl);
}

.welcome-content h2 {
    font-size: var(--font-size-3xl);
    font-weight: 700;
    color: var(--accent-color);
    margin-bottom: var(--spacing-md);
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.welcome-content p {
    font-size: var(--font-size-lg);
    color: var(--gray-300);
    max-width: 600px;
    margin: 0 auto;
}

/* Search Section */
.search-section {
    margin-bottom: var(--spacing-3xl);
}

.search-card {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-2xl);
    box-shadow: var(--shadow-xl);
    border: 2px solid var(--primary-color);
    backdrop-filter: blur(10px);
}

.search-header {
    text-align: center;
    margin-bottom: var(--spacing-xl);
}

.search-header h3 {
    font-size: var(--font-size-xl);
    font-weight: 600;
    color: var(--accent-color);
    margin-bottom: var(--spacing-sm);
}

.search-header h3 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.search-header p {
    color: var(--gray-300);
    font-size: var(--font-size-sm);
}

/* Form Grid */
.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-2xl);
}

.form-group {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
}

.form-group label {
    font-size: var(--font-size-sm);
    font-weight: 600;
    color: var(--white);
}

.form-group label i {
    color: var(--accent-color);
    margin-right: var(--spacing-xs);
}

.form-group input,
.form-group select {
    padding: var(--spacing-md);
    border: 2px solid var(--primary-color);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-base);
    transition: var(--transition-normal);
    background: var(--bg-darker);
    color: var(--white);
    box-shadow: var(--shadow-sm);
}

.form-group input:focus,
.form-group select:focus {
    outline: none;
    border-color: var(--accent-color);
    box-shadow: 0 0 0 3px rgba(0, 180, 216, 0.2);
    transform: translateY(-1px);
}

.form-group input::placeholder {
    color: var(--gray-400);
}

.input-info {
    font-size: var(--font-size-xs);
    color: var(--gray-400);
}

/* Scenario Selector */
.scenario-selector {
    background: var(--bg-darker);
    border-radius: var(--radius-lg);
    padding: var(--spacing-xl);
    margin-bottom: var(--spacing-xl);
    border: 2px solid var(--primary-dark);
    box-shadow: var(--shadow-lg);
}

.scenario-selector h4 {
    font-size: var(--font-size-lg);
    font-weight: 600;
    color: var(--accent-color);
    margin-bottom: var(--spacing-lg);
}

.scenario-selector h4 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.scenario-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-md);
}

/* Submit Container */
.submit-container {
    text-align: center;
    margin-top: var(--spacing-xl);
}

/* Buttons */
.btn-primary {
    background: var(--gradient-primary);
    color: var(--white);
    border: none;
    padding: var(--spacing-lg) var(--spacing-2xl);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-lg);
    font-weight: 600;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
    box-shadow: var(--shadow-lg);
    white-space: nowrap;
    border: 2px solid var(--accent-color);
    min-width: 250px;
    margin: 0 auto;
}

.btn-primary:hover:not(:disabled) {
    transform: translateY(-3px);
    box-shadow: var(--shadow-xl);
    border-color: var(--white);
}

.btn-primary:active {
    transform: translateY(-1px);
}

.btn-primary:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

.btn-secondary {
    background: var(--bg-card);
    color: var(--white);
    border: 2px solid var(--primary-color);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    box-shadow: var(--shadow-md);
}

.btn-secondary:hover {
    border-color: var(--accent-color);
    color: var(--accent-color);
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.table-action {
    padding: var(--spacing-xs) var(--spacing-sm);
    font-size: var(--font-size-xs);
    min-width: auto;
    margin: 0;
}

/* Results Section */
.results-section {
    margin-bottom: var(--spacing-3xl);
    display: none;
}

.results-section.show {
    display: block;
    animation: fadeInUp 0.8s ease-out;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-xl);
}

.results-header h3 {
    font-size: var(--font-size-xl);
    font-weight: 600;
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

/* Summary Cards */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-xl);
}

.summary-card {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    box-shadow: var(--shadow-xl);
    border-left: 4px solid;
    transition: var(--transition-normal);
    border: 2px solid var(--primary-dark);
    text-align: center;
}

.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-xl);
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
    font-size: var(--font-size-4xl);
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

.card-label {
    font-size: var(--font-size-sm);
    color: var(--gray-300);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

/* Scenario Info */
.scenario-info {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    margin-bottom: var(--spacing-xl);
    border: 2px solid var(--success-color);
    box-shadow: var(--shadow-xl);
}

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
}

.scenario-title {
    color: var(--white);
    font-size: var(--font-size-lg);
    font-weight: 600;
}

.severity-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.severity-badge.low {
    background: var(--success-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--success-color);
}

.severity-badge.medium {
    background: var(--warning-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--warning-color);
}

.severity-badge.high {
    background: var(--error-color);
    color: var(--white);
    box-shadow: 0 0 8px var(--error-color);
}

.scenario-content p {
    margin-bottom: var(--spacing-lg);
    color: var(--gray-300);
    font-size: var(--font-size-base);
}

.scenario-content strong {
    color: var(--white);
}

.action-btn {
    background: var(--gradient-primary);
    color: var(--white);
    border: none;
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: var(--transition-normal);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-right: var(--spacing-sm);
    margin-bottom: var(--spacing-sm);
    border: 1px solid var(--accent-color);
    box-shadow: var(--shadow-md);
}

.action-btn:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
    border-color: var(--white);
}

/* Comparison Tables */
.comparison-tables {
    display: grid;
    gap: var(--spacing-2xl);
    margin-bottom: var(--spacing-xl);
}

.table-section {
    background: var(--bg-card);
    border-radius: var(--radius-xl);
    box-shadow: var(--shadow-xl);
    overflow: hidden;
    border: 2px solid var(--primary-color);
}

.table-header {
    background: var(--gradient-primary);
    padding: var(--spacing-lg);
    text-align: center;
    border-bottom: 2px solid var(--accent-color);
}

.table-header h4 {
    color: var(--white);
    font-size: var(--font-size-lg);
    font-weight: 600;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
}

.table-header h4 i {
    margin-right: var(--spacing-sm);
    color: var(--accent-color);
}

.table-container {
    overflow-x: auto;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-darker);
}

.comparison-table th {
    background: var(--primary-dark);
    padding: var(--spacing-md);
    text-align: center;
    font-weight: 700;
    color: var(--white);
    border: 2px solid var(--primary-color);
    font-size: var(--font-size-sm);
    position: sticky;
    top: 0;
    z-index: 10;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.comparison-table td {
    padding: var(--spacing-md);
    text-align: center;
    border: 1px solid var(--primary-color);
    font-size: var(--font-size-sm);
    color: var(--white);
    vertical-align: middle;
    background: var(--bg-darker);
}

.comparison-table tbody tr:hover {
    background: var(--bg-card);
}

/* Column Styling with Enhanced Borders */
.oscar-col {
    background: rgba(26, 54, 93, 0.3) !important;
    border-left: 3px solid var(--primary-color) !important;
}

.copper-col {
    background: rgba(0, 180, 216, 0.3) !important;
    border-left: 3px solid var(--accent-color) !important;
}

.star-col {
    background: rgba(237, 137, 54, 0.3) !important;
    border-left: 3px solid var(--warning-color) !important;
}

.edb-col {
    background: rgba(56, 163, 105, 0.3) !important;
    border-left: 3px solid var(--success-color) !important;
}

.db-header {
    font-weight: 700 !important;
    font-size: var(--font-size-sm) !important;
    letter-spacing: 0.1em !important;
}

/* Status Badges - Much more readable and prominent */
.status-badge {
    padding: var(--spacing-sm) var(--spacing-md);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-xs);
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    border: 2px solid transparent;
    box-shadow: var(--shadow-md);
    min-width: 80px;
    display: inline-block;
    text-align: center;
}

.status-badge.status-match {
    background: var(--success-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 12px var(--success-color);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.status-badge.status-mismatch {
    background: var(--error-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 12px var(--error-color);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

.status-badge.status-missing {
    background: var(--warning-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 12px var(--warning-color);
    text-shadow: 0 1px 2px rgba(0, 0, 0, 0.5);
}

/* Enhanced visibility for other status badges */
.status-badge.found {
    background: var(--success-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 8px var(--success-color);
}

.status-badge.not-found {
    background: var(--error-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 8px var(--error-color);
}

.status-badge.error {
    background: var(--warning-color);
    color: var(--white);
    border-color: var(--white);
    box-shadow: 0 0 8px var(--warning-color);
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

    .comparison-table {
        font-size: var(--font-size-xs);
    }

    .comparison-table th,
    .comparison-table td {
        padding: var(--spacing-sm);
    }
}

@media (max-width: 480px) {
    .welcome-content h2 {
        font-size: var(--font-size-2xl);
    }

    .search-card {
        padding: var(--spacing-lg);
    }

    .summary-cards {
        grid-template-columns: 1fr;
    }
}

/* Utilities */
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
*:focus {
    outline: 2px solid var(--accent-color);
    outline-offset: 2px;
}

/* Scrollbar Styling */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-darker);
    border-radius: var(--radius-sm);
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: var(--radius-sm);
    border: 1px solid var(--accent-color);
}

::-webkit-scrollbar-thumb:hover {
    background: var(--primary-light);
}
