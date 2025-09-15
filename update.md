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
        <div class="header-content">
            <div class="logo">
                <i class="fas fa-exchange-alt"></i>
                <h1>OSCAR Reconcile</h1>
            </div>
            <div class="connection-status">
                <span class="status-dot" id="connection-status"></span>
                <span id="connection-text">Connected</span>
            </div>
        </div>
    </header>

    <!-- Main Content -->
    <main class="container">
        <!-- Welcome Section -->
        <section class="welcome-section">
            <div class="welcome-content">
                <h2>Data Reconciliation Between OSCAR, CoPPER, STAR & EDB</h2>
                <p>Compare and synchronize financial trading data across multiple systems</p>
            </div>
        </section>

        <!-- Form Section -->
        <section class="form-section">
            <div class="form-card">
                <div class="form-header">
                    <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                    <p>Configure your reconciliation criteria and comparison settings</p>
                </div>

                <form id="reconcile-form">
                    <div class="form-grid">
                        <div class="form-group">
                            <label for="reconcile-date">Reconciliation Date</label>
                            <input type="date" id="reconcile-date" name="reconcile-date" required>
                            <span class="input-info">Select the date for data comparison</span>
                        </div>

                        <div class="form-group">
                            <label for="guid">GUID (12 chars)</label>
                            <input type="text" id="guid" name="guid" placeholder="e.g., ABCDEFGH1234">
                            <span class="input-info">Global Unique Identifier</span>
                        </div>

                        <div class="form-group">
                            <label for="gfid">GFID (4 chars)</label>
                            <input type="text" id="gfid" name="gfid" placeholder="e.g., ABCD">
                            <span class="input-info">Globex Firm ID</span>
                        </div>

                        <div class="form-group">
                            <label for="gus-id">GUS ID (5 chars)</label>
                            <input type="text" id="gus-id" name="gus_id" placeholder="e.g., ABCDE">
                            <span class="input-info">Globex User Signature ID</span>
                        </div>

                        <div class="form-group">
                            <label for="contact-id">Contact ID</label>
                            <input type="text" id="contact-id" name="contact_id" placeholder="Contact Identifier">
                            <span class="input-info">Associated Contact Identifier</span>
                        </div>

                        <div class="form-group">
                            <label for="session-id">Session ID</label>
                            <input type="text" id="session-id" name="session_id" placeholder="e.g., MDBLZ, FIF">
                            <span class="input-info">Trading Session Identifier</span>
                        </div>
                    </div>

                    <div class="comparison-scenarios">
                        <h4>Comparison Scenarios</h4>
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

                    <button type="submit" class="execute-btn" id="submit-btn">
                        <i class="fas fa-sync-alt"></i>
                        Execute Reconciliation
                    </button>
                </form>
            </div>
        </section>

        <!-- Results Section -->
        <section class="results-section" id="results-section">
            <div class="results-header">
                <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                <div class="results-actions">
                    <button class="btn-secondary" id="export-btn">
                        <i class="fas fa-download"></i> Export Results
                    </button>
                    <button class="btn-secondary" id="refresh-btn">
                        <i class="fas fa-refresh"></i> Refresh Data
                    </button>
                    <button class="btn-secondary" id="clear-results-btn">
                        <i class="fas fa-times"></i> Clear Results
                    </button>
                </div>
            </div>

            <!-- Summary Cards -->
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

            <!-- Scenario Info -->
            <div class="scenario-info" id="scenario-info" style="display: none;">
                <div class="scenario-header">
                    <div class="scenario-title">SYNCHRONIZED: Data is synchronized between systems</div>
                    <div class="severity-badge">LOW</div>
                </div>
                <div class="scenario-content">
                    <p><strong>Input:</strong> <span id="scenario-input">TEST123</span> (<span id="scenario-type">GUID</span>)</p>
                    <p><strong>Status:</strong> <span id="scenario-status">Data found in both systems</span></p>
                    <div id="action-buttons">
                        <button class="action-btn">
                            <i class="fas fa-check"></i> No action required
                        </button>
                    </div>
                </div>
            </div>

            <!-- Comparison Tables -->
            <div class="comparison-tables">
                <!-- OSCAR ↔ CoPPER ↔ STAR Comparison -->
                <div class="table-container">
                    <div class="table-header">
                        <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ STAR Comparison</h4>
                    </div>
                    <div class="table-wrapper">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Record ID</th>
                                    <th class="oscar-col">GUID</th>
                                    <th class="oscar-col">Status</th>
                                    <th class="oscar-col">Last Updated</th>
                                    <th class="copper-col">GFID</th>
                                    <th class="copper-col">Status</th>
                                    <th class="copper-col">Session ID</th>
                                    <th class="star-col">Product ID</th>
                                    <th class="star-col">Status</th>
                                    <th class="star-col">Settlement</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>001</td>
                                    <td class="oscar-col">TESTGUID001</td>
                                    <td class="oscar-col">ACTIVE</td>
                                    <td class="oscar-col">2024-01-15</td>
                                    <td class="copper-col">TEST</td>
                                    <td class="copper-col">ACTIVE</td>
                                    <td class="copper-col">MDBLZ</td>
                                    <td class="star-col">ACTIVES</td>
                                    <td class="star-col">ACTIVE</td>
                                    <td class="star-col">COMPLETE</td>
                                    <td><span class="status-badge match">MATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-eye"></i></button></td>
                                </tr>
                                <tr>
                                    <td>002</td>
                                    <td class="oscar-col">TESTGUID002</td>
                                    <td class="oscar-col">EXPIRED</td>
                                    <td class="oscar-col">2023-12-31</td>
                                    <td class="copper-col">TEST</td>
                                    <td class="copper-col">ACTIVE</td>
                                    <td class="copper-col">FIF</td>
                                    <td class="star-col">-</td>
                                    <td class="star-col">-</td>
                                    <td class="star-col">-</td>
                                    <td><span class="status-badge mismatch">MISMATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-sync"></i></button></td>
                                </tr>
                                <tr>
                                    <td>003</td>
                                    <td class="oscar-col">TESTGUID003</td>
                                    <td class="oscar-col">ACTIVE</td>
                                    <td class="oscar-col">2024-01-10</td>
                                    <td class="copper-col">-</td>
                                    <td class="copper-col">-</td>
                                    <td class="copper-col">-</td>
                                    <td class="star-col">ACTIVES</td>
                                    <td class="star-col">PENDING</td>
                                    <td class="star-col">PENDING</td>
                                    <td><span class="status-badge missing">MISSING</span></td>
                                    <td><button class="action-icon"><i class="fas fa-plus"></i></button></td>
                                </tr>
                                <tr>
                                    <td>004</td>
                                    <td class="oscar-col">TESTGUID004</td>
                                    <td class="oscar-col">ACTIVE</td>
                                    <td class="oscar-col">2024-01-12</td>
                                    <td class="copper-col">TEST</td>
                                    <td class="copper-col">ACTIVE</td>
                                    <td class="copper-col">MDBLZ</td>
                                    <td class="star-col">ACTIVES</td>
                                    <td class="star-col">ACTIVE</td>
                                    <td class="star-col">COMPLETE</td>
                                    <td><span class="status-badge match">MATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-eye"></i></button></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- OSCAR ↔ CoPPER ↔ EDB Comparison -->
                <div class="table-container">
                    <div class="table-header">
                        <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER ↔ EDB Comparison</h4>
                    </div>
                    <div class="table-wrapper">
                        <table class="comparison-table">
                            <thead>
                                <tr>
                                    <th>Record ID</th>
                                    <th class="oscar-col">GUID</th>
                                    <th class="oscar-col">GUS ID</th>
                                    <th class="copper-col">Contact ID</th>
                                    <th class="copper-col">Session ID</th>
                                    <th class="copper-col">Product</th>
                                    <th class="copper-col">Permission</th>
                                    <th class="edb-col">Entity ID</th>
                                    <th class="edb-col">Type</th>
                                    <th class="edb-col">Schema</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>E001</td>
                                    <td class="oscar-col">TESTGUID005</td>
                                    <td class="oscar-col">GUS01</td>
                                    <td class="copper-col">CONT001</td>
                                    <td class="copper-col">MDBLZ</td>
                                    <td class="copper-col">BTEC_EU</td>
                                    <td class="copper-col">READ_WRITE</td>
                                    <td class="edb-col">ENT001</td>
                                    <td class="edb-col">USER</td>
                                    <td class="edb-col">TRADING</td>
                                    <td><span class="status-badge match">MATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-eye"></i></button></td>
                                </tr>
                                <tr>
                                    <td>E002</td>
                                    <td class="oscar-col">TESTGUID006</td>
                                    <td class="oscar-col">GUS02</td>
                                    <td class="copper-col">CONT002</td>
                                    <td class="copper-col">FIF</td>
                                    <td class="copper-col">EBS</td>
                                    <td class="copper-col">READ_ONLY</td>
                                    <td class="edb-col">-</td>
                                    <td class="edb-col">-</td>
                                    <td class="edb-col">-</td>
                                    <td><span class="status-badge missing">MISSING</span></td>
                                    <td><button class="action-icon"><i class="fas fa-plus"></i></button></td>
                                </tr>
                                <tr>
                                    <td>E003</td>
                                    <td class="oscar-col">TESTGUID007</td>
                                    <td class="oscar-col">GUS03</td>
                                    <td class="copper-col">CONT003</td>
                                    <td class="copper-col">MDBLZ</td>
                                    <td class="copper-col">CME_FO</td>
                                    <td class="copper-col">ADMIN</td>
                                    <td class="edb-col">ENT003</td>
                                    <td class="edb-col">ADMIN</td>
                                    <td class="edb-col">MANAGEMENT</td>
                                    <td><span class="status-badge mismatch">MISMATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-sync"></i></button></td>
                                </tr>
                                <tr>
                                    <td>E004</td>
                                    <td class="oscar-col">TESTGUID008</td>
                                    <td class="oscar-col">GUS04</td>
                                    <td class="copper-col">CONT004</td>
                                    <td class="copper-col">FIF</td>
                                    <td class="copper-col">BTEC_US</td>
                                    <td class="copper-col">READ_WRITE</td>
                                    <td class="edb-col">ENT004</td>
                                    <td class="edb-col">USER</td>
                                    <td class="edb-col">TRADING</td>
                                    <td><span class="status-badge match">MATCH</span></td>
                                    <td><button class="action-icon"><i class="fas fa-eye"></i></button></td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </section>
    </main>

    <script src="{{ url_for('static', filename='js/dynamic_javascript_original_ui.js') }}"></script>














    /* Modern OSCAR Reconciliation Tool Styles */
* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

:root {
    /* Blue Theme Colors */
    --primary-blue: #4A90E2;
    --primary-blue-dark: #357ABD;
    --primary-blue-light: #6BA3E8;
    
    /* Background Colors */
    --dark-bg: #2C3E50;
    --darker-bg: #1A252F;
    --card-bg: #34495E;
    --input-bg: #273746;
    
    /* Text Colors */
    --text-primary: #FFFFFF;
    --text-secondary: #BDC3C7;
    --text-muted: #95A5A6;
    
    /* Status Colors */
    --accent-green: #27AE60;
    --accent-red: #E74C3C;
    --accent-orange: #F39C12;
    --accent-yellow: #F1C40F;
    
    /* Border and Shadow */
    --border-color: #3D5A75;
    --border-light: #52708B;
    --shadow-primary: 0 8px 32px rgba(0,0,0,0.3);
    --shadow-card: 0 4px 20px rgba(0,0,0,0.2);
    
    /* Gradients - All Blue Theme */
    --gradient-header: linear-gradient(135deg, #4A90E2 0%, #357ABD 50%, #2E6DA4 100%);
    --gradient-button: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
    --gradient-bg: linear-gradient(135deg, #1A252F 0%, #2C3E50 100%);
    
    /* Spacing */
    --spacing-xs: 0.25rem;
    --spacing-sm: 0.5rem;
    --spacing-md: 1rem;
    --spacing-lg: 1.5rem;
    --spacing-xl: 2rem;
    --spacing-2xl: 3rem;
    
    /* Typography */
    --font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    --font-size-xs: 0.75rem;
    --font-size-sm: 0.875rem;
    --font-size-base: 1rem;
    --font-size-lg: 1.125rem;
    --font-size-xl: 1.25rem;
    --font-size-2xl: 1.5rem;
    --font-size-3xl: 2rem;
    
    /* Border Radius */
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 12px;
    --radius-xl: 15px;
    --radius-2xl: 20px;
    
    /* Transitions */
    --transition-fast: 0.2s ease;
    --transition-normal: 0.3s ease;
    --transition-slow: 0.5s ease;
}

/* Base Styles */
body {
    font-family: var(--font-family);
    background: var(--gradient-bg);
    color: var(--text-primary);
    min-height: 100vh;
    line-height: 1.6;
}

/* Header */
.header {
    background: var(--gradient-header);
    padding: var(--spacing-lg) 0;
    box-shadow: var(--shadow-primary);
    position: sticky;
    top: 0;
    z-index: 100;
}

.header-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 var(--spacing-xl);
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
    color: var(--text-primary);
}

.logo h1 {
    font-size: var(--font-size-2xl);
    font-weight: 700;
    color: var(--text-primary);
    margin: 0;
}

.connection-status {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    background: rgba(255,255,255,0.15);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: 25px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.1);
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
}

.connection-status span {
    font-size: var(--font-size-sm);
    font-weight: 500;
    color: var(--text-primary);
}

/* Main Container */
.container {
    max-width: 1200px;
    margin: 0 auto;
    padding: var(--spacing-xl);
}

/* Welcome Section */
.welcome-section {
    text-align: center;
    margin-bottom: var(--spacing-2xl);
}

.welcome-content h2 {
    font-size: var(--font-size-3xl);
    font-weight: 700;
    color: var(--primary-blue);
    margin-bottom: var(--spacing-md);
    line-height: 1.2;
}

.welcome-content p {
    color: var(--text-secondary);
    font-size: var(--font-size-lg);
    max-width: 600px;
    margin: 0 auto;
}

/* Form Card */
.form-card {
    background: var(--card-bg);
    border-radius: var(--radius-xl);
    padding: var(--spacing-2xl);
    margin-bottom: var(--spacing-2xl);
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-card);
}

.form-header {
    text-align: center;
    margin-bottom: var(--spacing-2xl);
}

.form-header h3 {
    color: var(--primary-blue);
    font-size: var(--font-size-xl);
    font-weight: 600;
    margin-bottom: var(--spacing-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
}

.form-header p {
    color: var(--text-secondary);
    font-size: var(--font-size-sm);
}

/* Form Grid */
.form-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-2xl);
}

.form-group {
    display: flex;
    flex-direction: column;
}

.form-group label {
    color: var(--text-primary);
    font-weight: 500;
    margin-bottom: var(--spacing-sm);
    font-size: var(--font-size-sm);
}

.form-group input, 
.form-group select {
    background: var(--input-bg);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-md);
    padding: var(--spacing-md);
    color: var(--text-primary);
    font-size: var(--font-size-sm);
    transition: all var(--transition-normal);
    font-family: var(--font-family);
}

.form-group input:focus, 
.form-group select:focus {
    outline: none;
    border-color: var(--primary-blue);
    box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.2);
    background: var(--darker-bg);
}

.form-group input::placeholder {
    color: var(--text-muted);
}

.input-info {
    font-size: var(--font-size-xs);
    color: var(--text-muted);
    margin-top: var(--spacing-xs);
}

/* Comparison Scenarios */
.comparison-scenarios {
    background: var(--input-bg);
    border-radius: var(--radius-lg);
    padding: var(--spacing-xl);
    margin-bottom: var(--spacing-2xl);
    border: 1px solid var(--border-light);
}

.comparison-scenarios h4 {
    color: var(--primary-blue);
    font-size: var(--font-size-lg);
    font-weight: 600;
    margin-bottom: var(--spacing-lg);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.scenario-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-lg);
}

/* Execute Button */
.execute-btn {
    background: var(--gradient-button);
    color: var(--text-primary);
    border: none;
    padding: var(--spacing-lg) var(--spacing-2xl);
    border-radius: var(--radius-lg);
    font-size: var(--font-size-base);
    font-weight: 600;
    cursor: pointer;
    transition: all var(--transition-normal);
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
    width: 100%;
    box-shadow: 0 4px 15px rgba(74, 144, 226, 0.3);
    font-family: var(--font-family);
}

.execute-btn:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(74, 144, 226, 0.4);
    background: linear-gradient(135deg, var(--primary-blue-light) 0%, var(--primary-blue) 100%);
}

.execute-btn:active {
    transform: translateY(0);
}

.execute-btn:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
}

/* Results Section */
.results-section {
    display: none;
}

.results-section.show {
    display: block;
}

.results-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-2xl);
}

.results-header h3 {
    color: var(--primary-blue);
    font-size: var(--font-size-xl);
    font-weight: 600;
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

.results-actions {
    display: flex;
    gap: var(--spacing-md);
}

.btn-secondary {
    background: var(--card-bg);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-md);
    font-size: var(--font-size-sm);
    cursor: pointer;
    transition: all var(--transition-normal);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    font-family: var(--font-family);
    font-weight: 500;
}

.btn-secondary:hover {
    background: var(--input-bg);
    transform: translateY(-1px);
    border-color: var(--primary-blue);
}

/* Summary Cards */
.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: var(--spacing-lg);
    margin-bottom: var(--spacing-2xl);
}

.summary-card {
    background: var(--card-bg);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    text-align: center;
    border-left: 4px solid;
    transition: all var(--transition-normal);
    border: 1px solid var(--border-color);
    position: relative;
    overflow: hidden;
}

.summary-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.1), transparent);
    opacity: 0;
    transition: opacity var(--transition-normal);
}

.summary-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-card);
}

.summary-card:hover::before {
    opacity: 1;
}

.summary-card.total { border-left-color: var(--primary-blue); }
.summary-card.matches { border-left-color: var(--accent-green); }
.summary-card.mismatches { border-left-color: var(--accent-red); }
.summary-card.missing { border-left-color: var(--accent-orange); }

.card-number {
    font-size: var(--font-size-3xl);
    font-weight: 700;
    margin-bottom: var(--spacing-sm);
    line-height: 1;
}

.total .card-number { color: var(--primary-blue); }
.matches .card-number { color: var(--accent-green); }
.mismatches .card-number { color: var(--accent-red); }
.missing .card-number { color: var(--accent-orange); }

.card-label {
    font-size: var(--font-size-sm);
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 500;
}

/* Scenario Info */
.scenario-info {
    background: var(--card-bg);
    border-radius: var(--radius-xl);
    padding: var(--spacing-xl);
    margin-bottom: var(--spacing-2xl);
    border: 1px solid var(--border-color);
    border-left: 4px solid var(--accent-green);
}

.scenario-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: var(--spacing-lg);
}

.scenario-title {
    color: var(--text-primary);
    font-size: var(--font-size-lg);
    font-weight: 600;
}

.severity-badge {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-2xl);
    font-size: var(--font-size-xs);
    font-weight: 600;
    text-transform: uppercase;
    background: var(--accent-green);
    color: var(--text-primary);
}

.severity-badge.medium {
    background: var(--accent-orange);
}

.severity-badge.high {
    background: var(--accent-red);
}

.scenario-content p {
    margin-bottom: var(--spacing-md);
    color: var(--text-secondary);
}

.action-btn {
    background: var(--primary-blue);
    color: var(--text-primary);
    border: none;
    padding: var(--spacing-sm) var(--spacing-lg);
    border-radius: var(--radius-md);
    font-size: var(--font-size-sm);
    font-weight: 500;
    cursor: pointer;
    transition: all var(--transition-normal);
    margin-right: var(--spacing-sm);
    margin-bottom: var(--spacing-sm);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    font-family: var(--font-family);
}

.action-btn:hover {
    background: var(--primary-blue-dark);
    transform: translateY(-1px);
}

/* Comparison Tables */
.comparison-tables {
    display: grid;
    gap: var(--spacing-2xl);
    margin-bottom: var(--spacing-2xl);
}

.table-container {
    background: var(--card-bg);
    border-radius: var(--radius-xl);
    overflow: hidden;
    border: 1px solid var(--border-color);
    box-shadow: var(--shadow-card);
}

.table-header {
    background: var(--primary-blue);
    padding: var(--spacing-lg);
    text-align: center;
}

.table-header h4 {
    color: var(--text-primary);
    font-size: var(--font-size-lg);
    font-weight: 600;
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: var(--spacing-sm);
}

.table-wrapper {
    overflow-x: auto;
}

.comparison-table {
    width: 100%;
    border-collapse: collapse;
    font-size: var(--font-size-sm);
}

.comparison-table th {
    background: var(--input-bg);
    padding: var(--spacing-md);
    text-align: center;
    font-weight: 600;
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    font-size: var(--font-size-xs);
    position: sticky;
    top: 0;
    z-index: 10;
}

.comparison-table td {
    padding: var(--spacing-md);
    text-align: center;
    border: 1px solid var(--border-color);
    color: var(--text-primary);
    font-size: var(--font-size-xs);
}

.comparison-table tbody tr {
    transition: background-color var(--transition-fast);
}

.comparison-table tbody tr:hover {
    background: var(--input-bg);
}

/* Column Styling */
.oscar-col { 
    background: rgba(74, 144, 226, 0.15); 
    border-left: 2px solid var(--primary-blue);
}

.copper-col { 
    background: rgba(39, 174, 96, 0.15); 
    border-left: 2px solid var(--accent-green);
}

.star-col { 
    background: rgba(241, 196, 15, 0.15); 
    border-left: 2px solid var(--accent-yellow);
}

.edb-col { 
    background: rgba(231, 76, 60, 0.15); 
    border-left: 2px solid var(--accent-red);
}

/* Status Badges */
.status-badge {
    padding: var(--spacing-xs) var(--spacing-sm);
    border-radius: var(--radius-xl);
    font-size: var(--font-size-xs);
    font-weight: 600;
    text-transform: uppercase;
    display: inline-block;
}

.status-badge.match {
    background: var(--accent-green);
    color: var(--text-primary);
}

.status-badge.mismatch {
    background: var(--accent-red);
    color: var(--text-primary);
}

.status-badge.missing {
    background: var(--accent-orange);
    color: var(--text-primary);
}

/* Action Icons */
.action-icon {
    background: var(--primary-blue);
    color: var(--text-primary);
    border: none;
    padding: var(--spacing-sm);
    border-radius: 50%;
    cursor: pointer;
    transition: all var(--transition-normal);
    font-size: var(--font-size-xs);
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
}

.action-icon:hover {
    background: var(--primary-blue-dark);
    transform: scale(1.1);
}

/* Loading States */
.loading {
    opacity: 0.6;
    pointer-events: none;
}

.spinner {
    width: 20px;
    height: 20px;
    border: 2px solid transparent;
    border-top: 2px solid currentColor;
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Responsive Design */
@media (max-width: 1024px) {
    .container {
        padding: var(--spacing-lg);
    }
    
    .form-grid {
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    }
    
    .scenario-grid {
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    }
}

@media (max-width: 768px) {
    .container {
        padding: var(--spacing-md);
    }
    
    .header-content {
        flex-direction: column;
        gap: var(--spacing-md);
        text-align: center;
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
        gap: var(--spacing-lg);
        align-items: stretch;
    }
    
    .results-actions {
        justify-content: center;
        flex-wrap: wrap;
    }
    
    .welcome-content h2 {
        font-size: var(--font-size-2xl);
    }
    
    .table-wrapper {
        font-size: var(--font-size-xs);
    }
    
    .comparison-table th,
    .comparison-table td {
        padding: var(--spacing-sm);
    }
}

@media (max-width: 480px) {
    .summary-cards {
        grid-template-columns: 1fr;
    }
    
    .form-card {
        padding: var(--spacing-lg);
    }
    
    .comparison-scenarios {
        padding: var(--spacing-lg);
    }
}

/* Accessibility */
@media (prefers-reduced-motion: reduce) {
    *,
    *::before,
    *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* Focus States */
button:focus-visible,
input:focus-visible,
select:focus-visible {
    outline: 2px solid var(--primary-blue);
    outline-offset: 2px;
}

/* Print Styles */
@media print {
    .header,
    .form-section,
    .results-actions {
        display: none;
    }
    
    .results-section {
        display: block !important;
    }
    
    .comparison-table {
        font-size: 10px;
    }
}





/* OSCAR Reconciliation Tool - Animations */

/* Keyframe Animations */
@keyframes fadeIn {
    from {
        opacity: 0;
    }
    to {
        opacity: 1;
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

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInFromLeft {
    from {
        opacity: 0;
        transform: translateX(-50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes slideInFromRight {
    from {
        opacity: 0;
        transform: translateX(50px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0;
        transform: scale(0.8);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

@keyframes pulse {
    0%, 100% {
        opacity: 1;
        transform: scale(1);
    }
    50% {
        opacity: 0.7;
        transform: scale(1.05);
    }
}

@keyframes bounce {
    0%, 20%, 53%, 80%, 100% {
        transform: translateY(0);
    }
    40%, 43% {
        transform: translateY(-10px);
    }
    70% {
        transform: translateY(-5px);
    }
    90% {
        transform: translateY(-2px);
    }
}

@keyframes spin {
    from {
        transform: rotate(0deg);
    }
    to {
        transform: rotate(360deg);
    }
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
    from {
        width: 0;
    }
    to {
        width: 100%;
    }
}

@keyframes blinkCursor {
    from, to {
        border-color: transparent;
    }
    50% {
        border-color: var(--primary-blue);
    }
}

/* Page Load Animations */
.welcome-section {
    animation: fadeInDown 0.8s ease-out;
}

.form-card {
    animation: fadeInUp 0.8s ease-out 0.2s both;
}

.results-section.show {
    animation: fadeInUp 0.6s ease-out;
}

/* Component Animations */
.summary-card {
    animation: scaleIn 0.5s ease-out;
}

.summary-card:nth-child(1) { animation-delay: 0.1s; }
.summary-card:nth-child(2) { animation-delay: 0.2s; }
.summary-card:nth-child(3) { animation-delay: 0.3s; }
.summary-card:nth-child(4) { animation-delay: 0.4s; }

.table-container {
    animation: fadeInUp 0.6s ease-out;
}

.table-container:nth-child(1) { animation-delay: 0.1s; }
.table-container:nth-child(2) { animation-delay: 0.2s; }

/* Interactive Animations */
.execute-btn {
    position: relative;
    overflow: hidden;
}

.execute-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.execute-btn:hover::before {
    left: 100%;
}

.btn-secondary {
    position: relative;
    overflow: hidden;
}

.btn-secondary::after {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    width: 0;
    height: 0;
    border-radius: 50%;
    background: rgba(74, 144, 226, 0.3);
    transform: translate(-50%, -50%);
    transition: width 0.6s, height 0.6s;
}

.btn-secondary:active::after {
    width: 300px;
    height: 300px;
}

/* Status Badge Animations */
.status-badge {
    position: relative;
    overflow: hidden;
}

.status-badge.match {
    animation: pulse 2s infinite;
}

.status-badge.mismatch {
    animation: bounce 1s ease-in-out;
}

.status-badge.missing {
    animation: pulse 1.5s infinite;
}

/* Form Input Animations */
.form-group input:focus,
.form-group select:focus {
    animation: glow 2s ease-in-out infinite;
}

.input-info {
    opacity: 0;
    transform: translateY(-10px);
    transition: all 0.3s ease;
}

.form-group:hover .input-info,
.form-group input:focus + .input-info,
.form-group select:focus + .input-info {
    opacity: 1;
    transform: translateY(0);
}

/* Connection Status Animation */
.status-dot {
    animation: pulse 2s infinite;
}

.status-dot.connected {
    animation: pulse 2s infinite;
}

.status-dot.disconnected {
    animation: bounce 1s infinite;
}

/* Table Row Animations */
.comparison-table tbody tr {
    opacity: 0;
    transform: translateX(-20px);
    animation: slideInFromLeft 0.5s ease-out forwards;
}

.comparison-table tbody tr:nth-child(1) { animation-delay: 0.1s; }
.comparison-table tbody tr:nth-child(2) { animation-delay: 0.2s; }
.comparison-table tbody tr:nth-child(3) { animation-delay: 0.3s; }
.comparison-table tbody tr:nth-child(4) { animation-delay: 0.4s; }
.comparison-table tbody tr:nth-child(5) { animation-delay: 0.5s; }

/* Action Icon Animations */
.action-icon {
    transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
}

.action-icon:hover {
    animation: bounce 0.6s ease;
}

/* Card Hover Effects */
.summary-card {
    transition: all 0.3s ease;
}

.summary-card:hover {
    animation: none;
    transform: translateY(-8px) scale(1.02);
}

/* Loading Animations */
.loading-spinner {
    animation: spin 1s linear infinite;
}

.loading-dots::after {
    content: '...';
    animation: typewriter 1.5s steps(3) infinite;
}

/* Shimmer Effect for Loading States */
.shimmer {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200px 100%;
    animation: shimmer 1.5s infinite;
}

/* Scenario Info Animation */
.scenario-info {
    animation: slideInFromLeft 0.6s ease-out;
}

.scenario-info .action-btn {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.4s ease-out forwards;
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
.progress-bar {
    width: 0%;
    height: 4px;
    background: var(--primary-blue);
    border-radius: 2px;
    transition: width 0.3s ease;
}

.progress-bar.loading {
    animation: shimmer 1.5s infinite;
}

/* Number Counter Animation */
.card-number {
    opacity: 0;
    animation: fadeIn 0.8s ease-out 0.5s forwards;
}

.card-number.counting {
    animation: bounce 0.6s ease-in-out;
}

/* Stagger Animation for Grid Items */
.form-group {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 0.5s ease-out forwards;
}

.form-group:nth-child(1) { animation-delay: 0.1s; }
.form-group:nth-child(2) { animation-delay: 0.2s; }
.form-group:nth-child(3) { animation-delay: 0.3s; }
.form-group:nth-child(4) { animation-delay: 0.4s; }
.form-group:nth-child(5) { animation-delay: 0.5s; }
.form-group:nth-child(6) { animation-delay: 0.6s; }

/* Header Animation */
.header {
    animation: slideInFromLeft 0.8s ease-out;
}

.logo {
    animation: fadeInDown 0.8s ease-out 0.2s both;
}

.connection-status {
    animation: fadeInDown 0.8s ease-out 0.4s both;
}

/* Elastic Scale Animation */
@keyframes elasticScale {
    0% {
        transform: scale(1);
    }
    20% {
        transform: scale(1.1);
    }
    40% {
        transform: scale(0.95);
    }
    60% {
        transform: scale(1.05);
    }
    80% {
        transform: scale(0.98);
    }
    100% {
        transform: scale(1);
    }
}

.execute-btn:active {
    animation: elasticScale 0.4s ease-out;
}

/* Floating Animation */
@keyframes float {
    0%, 100% {
        transform: translateY(0px);
    }
    50% {
        transform: translateY(-10px);
    }
}

.floating {
    animation: float 3s ease-in-out infinite;
}

/* Attention Seeking Animations */
@keyframes shake {
    0%, 100% {
        transform: translateX(0);
    }
    10%, 30%, 50%, 70%, 90% {
        transform: translateX(-5px);
    }
    20%, 40%, 60%, 80% {
        transform: translateX(5px);
    }
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
        transform: scale(1.05);
        box-shadow: 0 0 0 10px rgba(39, 174, 96, 0);
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
        transform: translateX(-5px);
    }
    75% {
        transform: translateX(5px);
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
    
    .status-dot {
        animation: none;
    }
    
    .floating {
        animation: none;
    }
    
    .pulse {
        animation: none;
    }
}

/* High Contrast Mode Support */
@media (prefers-contrast: high) {
    .shimmer {
        background: linear-gradient(90deg, #000 25%, #333 50%, #000 75%);
    }
}

/* Animation Utility Classes */
.animate-fade-in { animation: fadeIn 0.5s ease-out; }
.animate-fade-in-up { animation: fadeInUp 0.6s ease-out; }
.animate-fade-in-down { animation: fadeInDown 0.6s ease-out; }
.animate-slide-in-left { animation: slideInFromLeft 0.5s ease-out; }
.animate-slide-in-right { animation: slideInFromRight 0.5s ease-out; }
.animate-scale-in { animation: scaleIn 0.4s ease-out; }
.animate-bounce { animation: bounce 1s ease-in-out; }
.animate-pulse { animation: pulse 2s infinite; }
.animate-spin { animation: spin 1s linear infinite; }
.animate-float { animation: float 3s ease-in-out infinite; }
.animate-glow { animation: glow 2s ease-in-out infinite; }

/* Animation Delays */
.delay-100 { animation-delay: 0.1s; }
.delay-200 { animation-delay: 0.2s; }
.delay-300 { animation-delay: 0.3s; }
.delay-400 { animation-delay: 0.4s; }
.delay-500 { animation-delay: 0.5s; }

/* Animation Durations */
.duration-fast { animation-duration: 0.3s; }
.duration-normal { animation-duration: 0.5s; }
.duration-slow { animation-duration: 0.8s; }
</body>
</html>
