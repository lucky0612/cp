<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>OSCAR Reconciliation Tool - Standalone UI</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
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
            
            /* Gradients */
            --gradient-primary: linear-gradient(135deg, var(--primary-color) 0%, var(--primary-light) 100%);
            --gradient-success: linear-gradient(135deg, var(--success-color) 0%, #48bb78 100%);
            --gradient-warning: linear-gradient(135deg, var(--warning-color) 0%, #f6ad55 100%);
            --gradient-error: linear-gradient(135deg, var(--error-color) 0%, #fc8181 100%);
            
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
            max-width: 1400px;
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
            background: var(--success-color);
            animation: pulse 2s infinite;
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

        /* Search Section */
        .search-section {
            margin-bottom: var(--spacing-2xl);
        }

        .search-card {
            background: var(--white);
            border-radius: var(--radius-xl);
            padding: var(--spacing-2xl);
            box-shadow: var(--shadow-xl);
            border: 1px solid var(--gray-200);
        }

        .search-header {
            text-align: center;
            margin-bottom: var(--spacing-xl);
        }

        .search-header h3 {
            font-size: 1.25rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: var(--spacing-sm);
        }

        .search-header h3 i {
            margin-right: var(--spacing-sm);
            color: var(--accent-color);
        }

        /* Form Grid */
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

        .form-group label {
            font-size: 0.875rem;
            font-weight: 500;
            color: var(--gray-700);
        }

        .form-group input,
        .form-group select {
            padding: var(--spacing-md);
            border: 2px solid var(--gray-300);
            border-radius: var(--radius-lg);
            font-size: 1rem;
            transition: var(--transition-normal);
            background: var(--white);
        }

        .form-group input:focus,
        .form-group select:focus {
            outline: none;
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(26, 54, 93, 0.1);
        }

        .input-info {
            font-size: 0.75rem;
            color: var(--gray-500);
        }

        /* Scenario Selector */
        .scenario-selector {
            background: var(--gray-50);
            border-radius: var(--radius-lg);
            padding: var(--spacing-lg);
            margin-bottom: var(--spacing-xl);
        }

        .scenario-selector h4 {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--primary-color);
            margin-bottom: var(--spacing-md);
        }

        .scenario-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: var(--spacing-md);
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

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
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

        /* Submit Button Container */
        .submit-container {
            text-align: center;
            margin-top: var(--spacing-xl);
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

        /* Comparison Tables */
        .comparison-tables {
            display: grid;
            grid-template-columns: 1fr;
            gap: var(--spacing-2xl);
            margin-bottom: var(--spacing-xl);
        }

        .table-section {
            background: var(--white);
            border-radius: var(--radius-xl);
            box-shadow: var(--shadow-xl);
            overflow: hidden;
            border: 1px solid var(--gray-200);
        }

        .table-header {
            background: var(--gradient-primary);
            color: var(--white);
            padding: var(--spacing-lg);
            text-align: center;
        }

        .table-header h4 {
            font-size: 1.125rem;
            font-weight: 600;
            margin: 0;
        }

        .table-container {
            overflow-x: auto;
        }

        .comparison-table {
            width: 100%;
            border-collapse: collapse;
        }

        .comparison-table th {
            background: var(--gray-100);
            padding: var(--spacing-md);
            text-align: center;
            font-weight: 600;
            color: var(--gray-700);
            border-bottom: 2px solid var(--gray-200);
            position: sticky;
            top: 0;
            z-index: 10;
        }

        .comparison-table td {
            padding: var(--spacing-md);
            text-align: center;
            border-bottom: 1px solid var(--gray-200);
            vertical-align: middle;
        }

        .comparison-table tbody tr:hover {
            background: var(--gray-50);
        }

        /* Database Column Headers */
        .db-header {
            font-weight: 600;
            color: var(--white);
            background: var(--primary-color);
        }

        .oscar-col {
            background: rgba(26, 54, 93, 0.1);
            border-left: 3px solid var(--primary-color);
        }

        .copper-col {
            background: rgba(0, 180, 216, 0.1);
            border-left: 3px solid var(--accent-color);
        }

        .star-col {
            background: rgba(56, 161, 105, 0.1);
            border-left: 3px solid var(--success-color);
        }

        .edb-col {
            background: rgba(237, 137, 54, 0.1);
            border-left: 3px solid var(--warning-color);
        }

        /* Status Indicators */
        .status-badge {
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--radius-md);
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }

        .status-match {
            background: var(--success-color);
            color: var(--white);
        }

        .status-mismatch {
            background: var(--error-color);
            color: var(--white);
        }

        .status-missing {
            background: var(--warning-color);
            color: var(--white);
        }

        /* Summary Cards */
        .summary-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: var(--spacing-lg);
            margin-bottom: var(--spacing-xl);
        }

        .summary-card {
            background: var(--white);
            border-radius: var(--radius-xl);
            padding: var(--spacing-xl);
            box-shadow: var(--shadow-lg);
            border-left: 4px solid;
            transition: var(--transition-normal);
            text-align: center;
        }

        .summary-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-xl);
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

        .summary-card.total {
            border-left-color: var(--primary-color);
        }

        .card-number {
            font-size: 2.25rem;
            font-weight: 700;
            color: var(--primary-color);
            margin-bottom: var(--spacing-sm);
        }

        .card-label {
            font-size: 0.875rem;
            color: var(--gray-600);
            text-transform: uppercase;
            letter-spacing: 0.05em;
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

        .fade-in {
            animation: fadeInUp 0.8s ease-out;
        }

        .slide-up {
            animation: fadeInUp 0.8s ease-out;
        }

        /* Loading State */
        .loading {
            opacity: 0.6;
            pointer-events: none;
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
            
            .results-header {
                flex-direction: column;
                gap: var(--spacing-md);
                align-items: stretch;
            }
            
            .results-actions {
                justify-content: center;
            }
            
            .comparison-table {
                font-size: 0.875rem;
            }
            
            .comparison-table th,
            .comparison-table td {
                padding: var(--spacing-sm);
            }
        }

        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --white: #1a1a1a;
                --gray-50: #262626;
                --gray-100: #404040;
                --gray-200: #525252;
                --gray-300: #737373;
                --gray-400: #a3a3a3;
                --gray-500: #d4d4d4;
                --gray-600: #e5e5e5;
                --gray-700: #f5f5f5;
                --gray-800: #fafafa;
                --gray-900: #ffffff;
            }
            
            body {
                background: linear-gradient(135deg, #1a1a1a 0%, #262626 100%);
            }
        }

        /* Tooltip */
        .tooltip {
            position: relative;
            cursor: help;
        }

        .tooltip::after {
            content: attr(data-tooltip);
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: var(--gray-800);
            color: var(--white);
            padding: var(--spacing-xs) var(--spacing-sm);
            border-radius: var(--radius-sm);
            font-size: 0.75rem;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: var(--transition-normal);
        }

        .tooltip:hover::after {
            opacity: 1;
        }

        /* Hide initially */
        .results-section {
            display: none;
        }

        .results-section.show {
            display: block;
            animation: fadeInUp 0.6s ease-out;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <header class="header">
        <div class="container">
            <div class="header-content">
                <div class="logo">
                    <i class="fas fa-exchange-alt"></i>
                    <h1>OSCAR Reconcile</h1>
                </div>
                <div class="connection-status">
                    <div class="status-dot"></div>
                    <span>Connected</span>
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

            <!-- Search Form -->
            <section class="search-section">
                <div class="search-card slide-up">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Configure your reconciliation criteria and comparison settings</p>
                    </div>
                    
                    <form id="reconcile-form">
                        <!-- Date and Basic Inputs -->
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="reconcile-date">
                                    <i class="fas fa-calendar"></i> Reconciliation Date
                                </label>
                                <input type="date" id="reconcile-date" name="reconcile_date" required>
                                <div class="input-info">Select the date for data comparison</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="guid">
                                    <i class="fas fa-key"></i> GUID (12 chars)
                                </label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., ABCDEFGH1234" maxlength="12">
                                <div class="input-info">Global Unique Identifier</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gfid">
                                    <i class="fas fa-building"></i> GFID (4 chars)
                                </label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., ABCD" maxlength="4">
                                <div class="input-info">Globex Firm ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="gus-id">
                                    <i class="fas fa-user"></i> GUS ID (5 chars)
                                </label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., ABCDE" maxlength="5">
                                <div class="input-info">Globex User Signature ID</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="contact-id">
                                    <i class="fas fa-address-book"></i> Contact ID
                                </label>
                                <input type="text" id="contact-id" name="contact_id" placeholder="Contact identifier">
                                <div class="input-info">Associated contact identifier</div>
                            </div>
                            
                            <div class="form-group">
                                <label for="session-id">
                                    <i class="fas fa-plug"></i> Session ID
                                </label>
                                <input type="text" id="session-id" name="session_id" placeholder="e.g., MDBLZ, FIF">
                                <div class="input-info">Trading session identifier</div>
                            </div>
                        </div>

                        <!-- Scenario Selector -->
                        <div class="scenario-selector">
                            <h4><i class="fas fa-cogs"></i> Comparison Scenarios</h4>
                            <div class="scenario-grid">
                                <div class="form-group">
                                    <label for="comparison-type">Primary Comparison</label>
                                    <select id="comparison-type" name="comparison_type">
                                        <option value="guid_lookup">Standard GUID Lookup</option>
                                        <option value="scenario_2_1">Scenario 2.1 - Both Expired</option>
                                        <option value="scenario_2_2">Scenario 2.2 - OSCAR Expired, CoPPER Active</option>
                                        <option value="scenario_2_3">Scenario 2.3 - OSCAR Expired, CoPPER Missing</option>
                                        <option value="scenario_2_4">Scenario 2.4 - OSCAR Active, CoPPER Missing</option>
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
                                        <option value="both">Both STAR & EDB</option>
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

                        <!-- Submit Button -->
                        <div class="submit-container">
                            <button type="submit" class="btn-primary" id="submit-btn">
                                <i class="fas fa-sync-alt"></i>
                                <span>Execute Reconciliation</span>
                            </button>
                        </div>
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
                        <div class="card-number" id="total-records">247</div>
                        <div class="card-label">Total Records</div>
                    </div>
                    <div class="summary-card matches">
                        <div class="card-number" id="total-matches">189</div>
                        <div class="card-label">Matches</div>
                    </div>
                    <div class="summary-card mismatches">
                        <div class="card-number" id="total-mismatches">42</div>
                        <div class="card-label">Mismatches</div>
                    </div>
                    <div class="summary-card missing">
                        <div class="card-number" id="total-missing">16</div>
                        <div class="card-label">Missing Records</div>
                    </div>
                </div>

                <!-- Comparison Tables -->
                <div class="comparison-tables">
                    <!-- OSCAR to CoPPER to STAR -->
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
                                        <th class="copper-col">Status</th>
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
                                        <td class="copper-col">ACTIVE</td>
                                        <td class="copper-col">MDBLZ</td>
                                        <td class="star-col">ACTIVES</td>
                                        <td class="star-col">ACTIVE</td>
                                        <td class="star-col">COMPLETE</td>
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                        </td>
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
                                        <td><span class="status-badge status-mismatch">MISMATCH</span></td>
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-sync"></i>
                                            </button>
                                        </td>
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
                                        <td><span class="status-badge status-missing">MISSING</span></td>
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-plus"></i>
                                            </button>
                                        </td>
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
                                        <td><span class="status-badge status-match">MATCH</span></td>
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>

                    <!-- OSCAR to CoPPER to EDB -->
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
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                        </td>
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
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-plus"></i>
                                            </button>
                                        </td>
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
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-sync"></i>
                                            </button>
                                        </td>
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
                                        <td>
                                            <button class="btn-secondary" style="font-size: 0.75rem; padding: 0.25rem 0.5rem;">
                                                <i class="fas fa-eye"></i>
                                            </button>
                                        </td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </section>
        </div>
    </main>

    <script>
        // JavaScript for UI interactions
        document.addEventListener('DOMContentLoaded', function() {
            const form = document.getElementById('reconcile-form');
            const resultsSection = document.getElementById('results-section');
            const submitBtn = document.getElementById('submit-btn');
            
            // Set today's date as default
            const dateInput = document.getElementById('reconcile-date');
            const today = new Date().toISOString().split('T')[0];
            dateInput.value = today;
            
            // Form submission handler
            form.addEventListener('submit', function(e) {
                e.preventDefault();
                
                // Show loading state
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
                
                // Simulate processing delay
                setTimeout(() => {
                    // Show results
                    resultsSection.classList.add('show');
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                    
                    // Reset button
                    submitBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Execute Reconciliation';
                    submitBtn.disabled = false;
                    
                    // Update summary with random numbers
                    updateSummaryCards();
                }, 2000);
            });
            
            // Input validation
            const inputs = ['guid', 'gfid', 'gus-id', 'contact-id', 'session-id'];
            inputs.forEach(inputId => {
                const input = document.getElementById(inputId);
                if (input) {
                    input.addEventListener('input', function() {
                        validateInput(this);
                    });
                }
            });
            
            // Clear results
            document.getElementById('clear-results-btn').addEventListener('click', function() {
                resultsSection.classList.remove('show');
                form.reset();
                dateInput.value = today;
            });
            
            // Export functionality
            document.getElementById('export-btn').addEventListener('click', function() {
                const data = gatherTableData();
                downloadJSON(data, `oscar_reconciliation_${today}.json`);
            });
            
            // Refresh functionality
            document.getElementById('refresh-btn').addEventListener('click', function() {
                updateSummaryCards();
                showToast('Data refreshed successfully', 'success');
            });
        });
        
        function validateInput(input) {
            const value = input.value;
            const maxLength = input.maxLength;
            
            // Visual feedback for validation
            if (value.length === maxLength && maxLength > 0) {
                input.style.borderColor = 'var(--success-color)';
                input.style.boxShadow = '0 0 0 3px rgba(56, 161, 105, 0.1)';
            } else if (value.length > 0) {
                input.style.borderColor = 'var(--warning-color)';
                input.style.boxShadow = '0 0 0 3px rgba(237, 137, 54, 0.1)';
            } else {
                input.style.borderColor = 'var(--gray-300)';
                input.style.boxShadow = 'none';
            }
        }
        
        function updateSummaryCards() {
            const total = Math.floor(Math.random() * 100) + 200;
            const matches = Math.floor(total * 0.7) + Math.floor(Math.random() * 20);
            const mismatches = Math.floor(total * 0.2) + Math.floor(Math.random() * 10);
            const missing = total - matches - mismatches;
            
            document.getElementById('total-records').textContent = total;
            document.getElementById('total-matches').textContent = matches;
            document.getElementById('total-mismatches').textContent = mismatches;
            document.getElementById('total-missing').textContent = missing;
        }
        
        function gatherTableData() {
            const tables = document.querySelectorAll('.comparison-table');
            const data = {
                timestamp: new Date().toISOString(),
                reconciliation_date: document.getElementById('reconcile-date').value,
                parameters: {
                    guid: document.getElementById('guid').value,
                    gfid: document.getElementById('gfid').value,
                    gus_id: document.getElementById('gus-id').value,
                    contact_id: document.getElementById('contact-id').value,
                    session_id: document.getElementById('session-id').value,
                    comparison_type: document.getElementById('comparison-type').value,
                    comparison_field: document.getElementById('comparison-field').value
                },
                summary: {
                    total_records: document.getElementById('total-records').textContent,
                    matches: document.getElementById('total-matches').textContent,
                    mismatches: document.getElementById('total-mismatches').textContent,
                    missing: document.getElementById('total-missing').textContent
                },
                tables: []
            };
            
            tables.forEach((table, index) => {
                const tableData = {
                    name: index === 0 ? 'OSCAR_COPPER_STAR' : 'OSCAR_COPPER_EDB',
                    headers: [],
                    rows: []
                };
                
                // Get headers
                const headers = table.querySelectorAll('thead th');
                headers.forEach(header => {
                    tableData.headers.push(header.textContent.trim());
                });
                
                // Get data rows
                const rows = table.querySelectorAll('tbody tr');
                rows.forEach(row => {
                    const rowData = [];
                    const cells = row.querySelectorAll('td');
                    cells.forEach(cell => {
                        rowData.push(cell.textContent.trim());
                    });
                    tableData.rows.push(rowData);
                });
                
                data.tables.push(tableData);
            });
            
            return data;
        }
        
        function downloadJSON(data, filename) {
            const blob = new Blob([JSON.stringify(data, null, 2)], {
                type: 'application/json'
            });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
            
            showToast('Results exported successfully', 'success');
        }
        
        function showToast(message, type = 'info') {
            const toast = document.createElement('div');
            toast.className = `toast ${type}`;
            toast.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: white;
                padding: 1rem 1.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                border-left: 4px solid var(--${type === 'success' ? 'success' : 'primary'}-color);
                z-index: 1000;
                animation: slideInRight 0.3s ease-out;
            `;
            
            const colors = {
                success: 'var(--success-color)',
                error: 'var(--error-color)',
                warning: 'var(--warning-color)',
                info: 'var(--accent-color)'
            };
            
            toast.style.borderLeftColor = colors[type] || colors.info;
            toast.textContent = message;
            
            document.body.appendChild(toast);
            
            setTimeout(() => {
                toast.style.animation = 'slideOutRight 0.3s ease-out';
                setTimeout(() => {
                    document.body.removeChild(toast);
                }, 300);
            }, 3000);
        }
        
        // Add some CSS for toast animations
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
        `;
        document.head.appendChild(style);
    </script>
</body>
</html>
