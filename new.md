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
                    <h2>Data Reconciliation Between OSCAR & CoPPER</h2>
                    <p>Compare and synchronize financial trading data across systems</p>
                </div>
            </section>

            <!-- Search Section -->
            <section class="search-section slide-up">
                <div class="search-card">
                    <div class="search-header">
                        <h3><i class="fas fa-search"></i> Reconciliation Parameters</h3>
                        <p>Enter GUID alone OR both GFID and GUS ID together</p>
                    </div>

                    <form id="reconcile-form">
                        <div class="form-grid">
                            <div class="form-group">
                                <label for="guid"><i class="fas fa-key"></i> GUID</label>
                                <input type="text" id="guid" name="guid" placeholder="e.g., XFSH5YBIKCLZ">
                                <div class="input-info">Global Unique Identifier (enter alone)</div>
                            </div>
                            <div class="form-group">
                                <label for="gfid"><i class="fas fa-building"></i> GFID</label>
                                <input type="text" id="gfid" name="gfid" placeholder="e.g., PXCW">
                                <div class="input-info">Globex Firm ID (required with GUS ID)</div>
                            </div>
                            <div class="form-group">
                                <label for="gus-id"><i class="fas fa-user"></i> GUS ID</label>
                                <input type="text" id="gus-id" name="gus_id" placeholder="e.g., GJJ">
                                <div class="input-info">Globex User Signature ID (required with GFID)</div>
                            </div>
                        </div>
                        
                        <div class="scenario-selector">
                            <h4><i class="fas fa-cogs"></i> Comparison Options</h4>
                            <div class="scenario-grid">
                                <div class="form-group">
                                    <label for="comparison-type">Lookup Type</label>
                                    <select id="comparison-type" name="comparison_type">
                                        <option value="standard_lookup">Standard Lookup</option>
                                        <option value="scenario_2_1">Scenario 2.1 - Both Expired</option>
                                        <option value="scenario_2_2">Scenario 2.2 - OSCAR Expired, COPPER Active</option>
                                        <option value="scenario_2_3">Scenario 2.3 - OSCAR Expired, COPPER Missing</option>
                                        <option value="scenario_2_4">Scenario 2.4 - OSCAR Active, COPPER Missing</option>
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
            
            <!-- Results Section -->
            <section class="results-section" id="results-section">
                <div class="results-header">
                    <h3><i class="fas fa-chart-bar"></i> Reconciliation Results</h3>
                </div>

                <!-- Single Comparison Table -->
                <div class="comparison-tables">
                    <div class="table-section">
                        <div class="table-header">
                            <h4><i class="fas fa-database"></i> OSCAR ↔ CoPPER Comparison</h4>
                        </div>
                        <div class="table-container">
                            <table class="comparison-table">
                                <thead>
                                    <tr>
                                        <th colspan="4" class="db-header oscar-col">OSCAR</th>
                                        <th colspan="4" class="db-header copper-col">CoPPER</th>
                                        <th rowspan="2">Final Status</th>
                                        <th rowspan="2">Actions</th>
                                    </tr>
                                    <tr>
                                        <th class="oscar-col">GUID</th>
                                        <th class="oscar-col">GUS ID</th>
                                        <th class="oscar-col">GFID</th>
                                        <th class="oscar-col">Status</th>
                                        <th class="copper-col">GUID</th>
                                        <th class="copper-col">GUS ID</th>
                                        <th class="copper-col">GFID</th>
                                        <th class="copper-col">Status</th>
                                    </tr>
                                </thead>
                                <tbody id="comparison-table-body">
                                    <!-- Data will be populated dynamically -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>

                <!-- Scenario Information -->
                <div id="scenario-info-container" class="hidden">
                    <!-- Scenario details will be populated dynamically -->
                </div>
            </section>
        </div>
    </main>

    <script src="{{ url_for('static', filename='js/dynamic_javascript_original_ui.js') }}"></script>
</body>
</html>

















// Dynamic JavaScript for OSCAR Reconciliation Tool
// Corrected version with proper table structure and action handling

document.addEventListener("DOMContentLoaded", function() {
    // Get DOM elements
    const form = document.getElementById("reconcile-form");
    const resultsSection = document.getElementById("results-section");
    const submitBtn = document.getElementById("submit-btn");
    const tableBody = document.getElementById("comparison-table-body");
    const scenarioContainer = document.getElementById("scenario-info-container");
    
    // Input fields
    const guidInput = document.getElementById("guid");
    const gfidInput = document.getElementById("gfid");
    const gusInput = document.getElementById("gus-id");

    // Initialize system
    checkSystemHealth();
    
    // Check system health every 5 minutes
    setInterval(checkSystemHealth, 300000);

    // Form submission handler
    if (form) {
        form.addEventListener("submit", function(e) {
            e.preventDefault();
            executeReconciliation();
        });
    }

    // Input validation - ensure GFID and GUS are together
    guidInput.addEventListener('input', function() {
        if (this.value.trim()) {
            gfidInput.disabled = true;
            gusInput.disabled = true;
            gfidInput.value = '';
            gusInput.value = '';
        } else {
            gfidInput.disabled = false;
            gusInput.disabled = false;
        }
    });

    [gfidInput, gusInput].forEach(input => {
        input.addEventListener('input', function() {
            if (gfidInput.value.trim() || gusInput.value.trim()) {
                guidInput.disabled = true;
                guidInput.value = '';
            } else {
                guidInput.disabled = false;
            }
        });
    });

    async function checkSystemHealth() {
        try {
            const response = await fetch('/api/health_check');
            const healthData = await response.json();
            
            updateConnectionStatus(healthData);
            
        } catch (error) {
            console.error('Health check failed:', error);
            updateConnectionStatus({ 
                status: 'unhealthy', 
                database_status: { oscar: false, copper: false } 
            });
        }
    }

    function updateConnectionStatus(healthData) {
        const statusDot = document.getElementById('connection-status');
        const statusText = document.getElementById('connection-text');
        
        if (statusDot && statusText) {
            if (healthData.status === 'healthy') {
                statusDot.className = 'status-dot connected';
                statusText.textContent = 'Connected';
            } else {
                statusDot.className = 'status-dot';
                statusText.textContent = 'Degraded';
            }
        }
    }

    async function executeReconciliation() {
        // Show loading state
        setLoadingState(true);
        
        try {
            // Get form data
            const formData = getFormData();
            
            // Validate form data
            const validation = validateFormData(formData);
            if (!validation.valid) {
                showToast(validation.message, 'error');
                setLoadingState(false);
                return;
            }

            // Execute reconciliation
            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            const result = await response.json();

            if (result.success) {
                // Store result for reference
                window.lastReconciliationData = result;
                
                // Update UI with results
                updateResultsDisplay(result);
                
                // Show results section
                if (resultsSection) {
                    resultsSection.classList.add('show');
                    resultsSection.scrollIntoView({ behavior: 'smooth' });
                }
                
                showToast('Reconciliation completed successfully', 'success');
            } else {
                showToast(result.error || 'Reconciliation failed', 'error');
            }

        } catch (error) {
            console.error('Reconciliation error:', error);
            showToast('Network error - please check connection', 'error');
        } finally {
            setLoadingState(false);
        }
    }

    function getFormData() {
        return {
            guid: guidInput?.value?.trim() || '',
            gfid: gfidInput?.value?.trim() || '',
            gus_id: gusInput?.value?.trim() || '',
            comparison_type: document.getElementById('comparison-type')?.value || 'standard_lookup'
        };
    }

    function validateFormData(data) {
        // Check validation rules
        if (!data.guid && !data.gfid && !data.gus_id) {
            return {
                valid: false,
                message: 'Please provide either GUID alone OR both GFID and GUS ID together'
            };
        }

        if (data.guid && (data.gfid || data.gus_id)) {
            return {
                valid: false,
                message: 'Please provide either GUID alone OR both GFID and GUS ID together'
            };
        }

        if ((data.gfid && !data.gus_id) || (data.gus_id && !data.gfid)) {
            return {
                valid: false,
                message: 'GFID and GUS ID must be provided together'
            };
        }

        return { valid: true, message: 'Valid' };
    }

    function updateResultsDisplay(result) {
        // Clear existing results
        tableBody.innerHTML = '';
        scenarioContainer.innerHTML = '';
        scenarioContainer.classList.add('hidden');

        // Create table row
        const row = createTableRow(result);
        tableBody.appendChild(row);

        // Show scenario information
        if (result.scenario && result.scenario.recommended_actions.length > 0) {
            createScenarioInfo(result);
        }
    }

    function createTableRow(result) {
        const row = document.createElement('tr');
        
        // Extract data from result
        const oscarData = result.oscar_data.record || {};
        const copperData = result.copper_data.record || {};
        const scenario = result.scenario || {};
        
        // OSCAR columns
        const oscarGuid = oscarData.guid || (result.oscar_data.found ? 'N/A' : '-');
        const oscarGusId = oscarData.gus_id || (result.oscar_data.found ? 'N/A' : '-');
        const oscarGfid = oscarData.gfid || (result.oscar_data.found ? 'N/A' : '-');
        const oscarStatus = result.oscar_data.status_with_date || 'MISSING';
        
        // COPPER columns
        const copperGuid = copperData.guid || (result.copper_data.found ? 'N/A' : '-');
        const copperGusId = copperData.gus_id || (result.copper_data.found ? 'N/A' : '-');
        const copperGfid = copperData.gfid || (result.copper_data.found ? 'N/A' : '-');
        const copperStatus = result.copper_data.status_with_date || 'MISSING';
        
        // Final status
        const finalStatus = scenario.final_status || 'UNKNOWN';
        
        row.innerHTML = `
            <td class="oscar-col">${oscarGuid}</td>
            <td class="oscar-col">${oscarGusId}</td>
            <td class="oscar-col">${oscarGfid}</td>
            <td class="oscar-col">${oscarStatus}</td>
            <td class="copper-col">${copperGuid}</td>
            <td class="copper-col">${copperGusId}</td>
            <td class="copper-col">${copperGfid}</td>
            <td class="copper-col">${copperStatus}</td>
            <td>${createStatusBadge(finalStatus)}</td>
            <td>${createActionButton(scenario, result.input_value)}</td>
        `;
        
        return row;
    }

    function createStatusBadge(status) {
        let badgeClass = 'status-badge';
        let icon = '';
        
        switch (status.toLowerCase()) {
            case 'match':
                badgeClass += ' status-match';
                icon = '<i class="fas fa-check"></i>';
                break;
            case 'mismatch':
                badgeClass += ' status-mismatch';
                icon = '<i class="fas fa-exclamation-triangle"></i>';
                break;
            case 'missing':
                badgeClass += ' status-missing';
                icon = '<i class="fas fa-question-circle"></i>';
                break;
            default:
                badgeClass += ' status-missing';
                icon = '<i class="fas fa-question"></i>';
                break;
        }
        
        return `<span class="${badgeClass}">${icon} ${status.toUpperCase()}</span>`;
    }

    function createActionButton(scenario, inputValue) {
        if (!scenario.recommended_actions || scenario.recommended_actions.length === 0) {
            return '<button class="btn-secondary table-action" disabled><i class="fas fa-check"></i></button>';
        }
        
        // Get primary action
        const primaryAction = scenario.recommended_actions[0];
        let icon = '';
        
        // Determine icon based on action type
        if (primaryAction.includes('Sync Job') || primaryAction.includes('sync')) {
            icon = 'fas fa-sync-alt';
        } else if (primaryAction.includes('Sync Flag')) {
            icon = 'fas fa-flag';
        } else if (primaryAction.includes('Investigate') || primaryAction.includes('investigation')) {
            icon = 'fas fa-search';
        } else if (primaryAction.includes('Verify')) {
            icon = 'fas fa-check-circle';
        } else if (primaryAction.includes('Manual')) {
            icon = 'fas fa-hand-paper';
        } else {
            icon = 'fas fa-cog';
        }
        
        return `<button class="btn-secondary table-action" onclick="showActionMenu('${inputValue}', '${scenario.type}')" title="${primaryAction}">
                    <i class="${icon}"></i>
                </button>`;
    }

    // Make showActionMenu globally available
    window.showActionMenu = function(inputValue, scenarioType) {
        const result = window.lastReconciliationData;
        if (result && result.scenario) {
            showActionModal(result.scenario.recommended_actions, inputValue);
        }
    };

    function showActionModal(actions, inputValue) {
        const modal = document.createElement('div');
        modal.className = 'action-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-cog"></i> Available Actions</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <p><strong>For:</strong> ${inputValue}</p>
                    <div class="actions-list">
                        ${actions.map(action => `
                            <button class="action-btn" onclick="executeAction('${action}', '${inputValue}')">
                                <i class="fas fa-${getActionIcon(action)}"></i> ${action}
                            </button>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Close modal functionality
        const closeBtn = modal.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    function getActionIcon(action) {
        if (action.includes('Sync Job')) return 'sync-alt';
        if (action.includes('Sync Flag')) return 'flag';
        if (action.includes('Investigate')) return 'search';
        if (action.includes('Verify')) return 'check-circle';
        if (action.includes('Manual')) return 'hand-paper';
        return 'cog';
    }

    // Make executeAction globally available
    window.executeAction = async function(action, inputValue) {
        try {
            // Close modal
            const modal = document.querySelector('.action-modal');
            if (modal) {
                document.body.removeChild(modal);
            }
            
            setLoadingState(true, `Executing ${action}...`);
            
            const response = await fetch('/api/execute_action', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: action,
                    input_value: inputValue
                })
            });

            const result = await response.json();

            if (result.success) {
                showActionResult(result);
                showToast(result.message || `${action} completed successfully`, 'success');
            } else {
                showToast(result.error || `${action} failed`, 'error');
            }

        } catch (error) {
            console.error('Action execution error:', error);
            showToast(`Failed to execute ${action}`, 'error');
        } finally {
            setLoadingState(false);
        }
    };

    function createScenarioInfo(result) {
        const scenario = result.scenario;
        
        scenarioContainer.innerHTML = `
            <div class="scenario-card">
                <div class="scenario-header">
                    <h4>${scenario.type}: ${scenario.description}</h4>
                    <span class="severity-badge ${scenario.severity.toLowerCase()}">${scenario.severity}</span>
                </div>
                <div class="scenario-content">
                    <p><strong>Input:</strong> ${result.input_value} (${result.input_type})</p>
                    <p><strong>OSCAR Status:</strong> ${result.oscar_data.status_with_date}</p>
                    <p><strong>Copper Status:</strong> ${result.copper_data.status_with_date}</p>
                    <div class="recommended-actions">
                        <h5>Recommended Actions:</h5>
                        <div class="actions-buttons">
                            ${scenario.recommended_actions.map(action => `
                                <button class="action-btn" onclick="executeAction('${action}', '${result.input_value}')">
                                    <i class="fas fa-${getActionIcon(action)}"></i> ${action}
                                </button>
                            `).join('')}
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        scenarioContainer.classList.remove('hidden');
    }

    function showActionResult(result) {
        const resultModal = document.createElement('div');
        resultModal.className = 'action-result-modal';
        resultModal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-check-circle"></i> Action Completed</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <h4>${result.action}</h4>
                    <p><strong>Status:</strong> ${result.message}</p>
                    ${result.details ? `<p><strong>Details:</strong> ${result.details}</p>` : ''}
                    ${result.execution_time ? `<p><strong>Execution Time:</strong> ${result.execution_time}</p>` : ''}
                    ${result.records_processed ? `<p><strong>Records Processed:</strong> ${result.records_processed}</p>` : ''}
                    ${result.records_updated ? `<p><strong>Records Updated:</strong> ${result.records_updated}</p>` : ''}
                    ${result.ticket_id ? `<p><strong>Ticket ID:</strong> ${result.ticket_id}</p>` : ''}
                </div>
            </div>
        `;

        document.body.appendChild(resultModal);

        // Close modal functionality
        const closeBtn = resultModal.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(resultModal);
        });

        // Auto-close after 8 seconds
        setTimeout(() => {
            if (document.body.contains(resultModal)) {
                document.body.removeChild(resultModal);
            }
        }, 8000);
    }

    function setLoadingState(loading, message = 'Processing reconciliation...') {
        if (submitBtn) {
            if (loading) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
            } else {
                submitBtn.innerHTML = '<i class="fas fa-sync-alt"></i> Execute Reconciliation';
                submitBtn.disabled = false;
            }
        }
    }

    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas fa-${getToastIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;
        
        // Style the toast
        Object.assign(toast.style, {
            position: 'fixed',
            top: '20px',
            right: '20px',
            background: 'white',
            color: 'var(--gray-800)',
            padding: '1rem 1.5rem',
            borderRadius: '0.5rem',
            boxShadow: '0 10px 25px rgba(0, 0, 0, 0.1)',
            zIndex: '1000',
            borderLeft: `4px solid ${getToastColor(type)}`,
            maxWidth: '400px',
            animation: 'slideInRight 0.3s ease-out'
        });
        
        document.body.appendChild(toast);
        
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease-out';
            setTimeout(() => {
                if (document.body.contains(toast)) {
                    document.body.removeChild(toast);
                }
            }, 300);
        }, 4000);
    }

    function getToastIcon(type) {
        const icons = {
            success: 'check-circle',
            error: 'exclamation-circle',
            warning: 'exclamation-triangle',
            info: 'info-circle'
        };
        return icons[type] || icons.info;
    }

    function getToastColor(type) {
        const colors = {
            success: '#38a169',
            error: '#e53e3e',
            warning: '#ed8936',
            info: '#00b4d8'
        };
        return colors[type] || colors.info;
    }

    // Add CSS for new components
    if (!document.querySelector('#dynamic-styles')) {
        const style = document.createElement('style');
        style.id = 'dynamic-styles';
        style.textContent = `
            .hidden { display: none !important; }
            
            .scenario-card {
                background: var(--bg-card);
                border-radius: var(--radius-xl);
                padding: var(--spacing-xl);
                box-shadow: var(--shadow-xl);
                border-left: 4px solid var(--accent-color);
                margin-top: var(--spacing-xl);
            }
            
            .scenario-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: var(--spacing-lg);
            }
            
            .scenario-header h4 {
                color: var(--white);
                font-size: var(--font-size-lg);
                font-weight: 600;
                margin: 0;
            }
            
            .severity-badge {
                padding: var(--spacing-xs) var(--spacing-md);
                border-radius: var(--radius-lg);
                font-size: var(--font-size-xs);
                font-weight: 700;
                text-transform: uppercase;
            }
            
            .severity-badge.low { background: var(--success-color); color: white; }
            .severity-badge.medium { background: var(--warning-color); color: white; }
            .severity-badge.high { background: var(--error-color); color: white; }
            
            .scenario-content p {
                margin-bottom: var(--spacing-sm);
                color: var(--gray-300);
            }
            
            .scenario-content strong {
                color: var(--white);
            }
            
            .recommended-actions h5 {
                margin: var(--spacing-lg) 0 var(--spacing-sm) 0;
                color: var(--white);
            }
            
            .actions-buttons {
                display: flex;
                flex-wrap: wrap;
                gap: var(--spacing-sm);
            }
            
            .action-btn {
                background: var(--gradient-primary);
                color: white;
                border: none;
                padding: var(--spacing-sm) var(--spacing-lg);
                border-radius: var(--radius-lg);
                font-size: var(--font-size-sm);
                font-weight: 500;
                cursor: pointer;
                transition: all 0.3s ease;
                display: flex;
                align-items: center;
                gap: var(--spacing-sm);
                border: 1px solid var(--accent-color);
            }
            
            .action-btn:hover {
                transform: translateY(-2px);
                box-shadow: var(--shadow-md);
            }
            
            .action-modal, .action-result-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.7);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
            }
            
            .modal-content {
                background: var(--bg-card);
                border-radius: var(--radius-xl);
                max-width: 500px;
                width: 90%;
                max-height: 80vh;
                overflow: auto;
                box-shadow: var(--shadow-xl);
                border: 2px solid var(--primary-color);
            }
            
            .modal-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: var(--spacing-xl);
                border-bottom: 1px solid var(--primary-color);
            }
            
            .modal-header h3 {
                margin: 0;
                color: var(--white);
                font-size: var(--font-size-xl);
            }
            
            .modal-close {
                background: none;
