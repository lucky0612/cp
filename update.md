// Dynamic JavaScript for OSCAR Reconciliation Tool
// Simplified OSCAR-Copper reconciliation with validation

document.addEventListener("DOMContentLoaded", function() {
    // Get DOM elements
    const form = document.getElementById("reconcile-form");
    const resultsSection = document.getElementById("results-section");
    const submitBtn = document.getElementById("submit-btn");
    
    // Input elements
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

    // Input validation listeners
    if (guidInput) {
        guidInput.addEventListener('input', validateInputs);
    }
    if (gfidInput) {
        gfidInput.addEventListener('input', validateInputs);
    }
    if (gusInput) {
        gusInput.addEventListener('input', validateInputs);
    }

    function validateInputs() {
        const guid = guidInput ? guidInput.value.trim() : '';
        const gfid = gfidInput ? gfidInput.value.trim() : '';
        const gus = gusInput ? gusInput.value.trim() : '';

        // Clear previous validation styles
        clearValidationStyles();

        // Validation logic: GUID alone OR both GFID+GUS together
        if (guid && (gfid || gus)) {
            showValidationError('Please provide either GUID alone OR both GFID and GUS ID together');
            return false;
        }

        if (!guid && (gfid && !gus)) {
            showValidationError('GUS ID is required when GFID is provided');
            highlightInput(gusInput, 'error');
            return false;
        }

        if (!guid && (!gfid && gus)) {
            showValidationError('GFID is required when GUS ID is provided');
            highlightInput(gfidInput, 'error');
            return false;
        }

        if (!guid && !gfid && !gus) {
            showValidationError('Please provide either GUID alone OR both GFID and GUS ID together');
            return false;
        }

        // Valid input - show success styling
        if (guid) {
            highlightInput(guidInput, 'success');
        } else if (gfid && gus) {
            highlightInput(gfidInput, 'success');
            highlightInput(gusInput, 'success');
        }

        clearValidationError();
        return true;
    }

    function clearValidationStyles() {
        [guidInput, gfidInput, gusInput].forEach(input => {
            if (input) {
                input.style.borderColor = '';
                input.style.boxShadow = '';
            }
        });
    }

    function highlightInput(input, type) {
        if (!input) return;
        
        if (type === 'success') {
            input.style.borderColor = 'var(--success-color)';
            input.style.boxShadow = '0 0 0 3px rgba(56, 163, 165, 0.2)';
        } else if (type === 'error') {
            input.style.borderColor = 'var(--error-color)';
            input.style.boxShadow = '0 0 0 3px rgba(229, 62, 62, 0.2)';
        }
    }

    function showValidationError(message) {
        clearValidationError();
        
        const errorDiv = document.createElement('div');
        errorDiv.id = 'validation-error';
        errorDiv.className = 'validation-error';
        errorDiv.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${message}`;
        
        const submitContainer = document.querySelector('.submit-container');
        if (submitContainer) {
            submitContainer.parentNode.insertBefore(errorDiv, submitContainer);
        }
    }

    function clearValidationError() {
        const errorDiv = document.getElementById('validation-error');
        if (errorDiv) {
            errorDiv.remove();
        }
    }

    async function checkSystemHealth() {
        try {
            console.log('Checking system health...');
            const response = await fetch('/api/health_check', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            
            console.log('Health check response status:', response.status);
            const healthData = await response.json();
            console.log('Health check data:', healthData);
            
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
                console.log('System status: Connected');
            } else {
                statusDot.className = 'status-dot disconnected';
                statusText.textContent = 'Degraded';
                console.log('System status: Degraded');
            }
        }
    }

    async function executeReconciliation() {
        console.log('Starting reconciliation...');
        
        // Validate inputs
        if (!validateInputs()) {
            console.log('Validation failed');
            return;
        }

        // Show loading state
        setLoadingState(true);
        
        try {
            // Get form data
            const formData = getFormData();
            console.log('Form data:', formData);
            
            // Execute reconciliation
            console.log('Sending reconciliation request...');
            const response = await fetch('/api/reconcile', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(formData)
            });

            console.log('Reconciliation response status:', response.status);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const result = await response.json();
            console.log('Reconciliation result:', result);

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
            showToast(`Network error: ${error.message}`, 'error');
        } finally {
            setLoadingState(false);
        }
    }

    function getFormData() {
        return {
            guid: guidInput ? guidInput.value.trim() : '',
            gfid: gfidInput ? gfidInput.value.trim() : '',
            gus_id: gusInput ? gusInput.value.trim() : '',
            comparison_type: document.getElementById('comparison-type')?.value || 'guid_lookup',
            comparison_field: document.getElementById('comparison-field')?.value || 'status'
        };
    }

    function updateResultsDisplay(result) {
        console.log('Updating results display:', result);
        
        // Update scenario information
        updateScenarioInfo(result);
        
        // Update table with results
        updateComparisonTable(result);
    }

    function updateScenarioInfo(result) {
        // Remove existing scenario info
        const existingScenario = document.getElementById('scenario-info-container');
        if (existingScenario) {
            existingScenario.innerHTML = '';
        }

        if (result.scenario) {
            const scenarioHtml = `
                <div class="scenario-info">
                    <div class="scenario-header">
                        <div class="scenario-title">${result.scenario.type}: ${result.scenario.description}</div>
                        <div class="severity-badge ${result.scenario.severity.toLowerCase()}">${result.scenario.severity}</div>
                    </div>
                    <div class="scenario-content">
                        <p><strong>Input:</strong> ${result.input_value} (${result.input_type})</p>
                        <p><strong>OSCAR Status:</strong> ${result.oscar_data.status_with_date || 'Not found'}</p>
                        <p><strong>Copper Status:</strong> ${result.copper_data.status_with_date || 'Not found'}</p>
                        ${result.scenario.recommended_actions && result.scenario.recommended_actions.length > 0 ? `
                            <div class="recommended-actions">
                                <strong>Recommended Actions:</strong>
                                ${result.scenario.recommended_actions.map(action => 
                                    `<button class="action-btn" onclick="executeAction('${action}', '${result.input_value}')">${action}</button>`
                                ).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            if (existingScenario) {
                existingScenario.innerHTML = scenarioHtml;
            }
        }
    }

    function updateComparisonTable(result) {
        const table = document.querySelector('.comparison-table tbody');
        if (!table) {
            console.log('Table body not found');
            return;
        }

        // Clear existing rows
        table.innerHTML = '';

        // Create new row with actual data
        const row = document.createElement('tr');
        
        const oscarRecord = result.oscar_data.record || {};
        const copperRecord = result.copper_data.record || {};
        
        // Get status for badge
        const finalStatus = result.scenario.final_status || 'UNKNOWN';
        const statusBadgeClass = getStatusBadgeClass(finalStatus);

        row.innerHTML = `
            <td>${result.input_value}</td>
            <td class="oscar-col">${oscarRecord.guid || result.input_value || '-'}</td>
            <td class="oscar-col">${oscarRecord.gfid || '-'}</td>
            <td class="oscar-col">${oscarRecord.gus_id || '-'}</td>
            <td class="oscar-col">${result.oscar_data.status_with_date || 'MISSING'}</td>
            <td class="copper-col">${copperRecord.guid || '-'}</td>
            <td class="copper-col">${copperRecord.gfid || '-'}</td>
            <td class="copper-col">${copperRecord.gus_id || '-'}</td>
            <td class="copper-col">${result.copper_data.status_with_date || 'MISSING'}</td>
            <td><span class="status-badge ${statusBadgeClass}">${finalStatus}</span></td>
            <td><button class="btn-secondary table-action" onclick="showDetails('${result.input_value}')"><i class="fas fa-eye"></i></button></td>
        `;

        table.appendChild(row);
    }

    function getStatusBadgeClass(status) {
        switch (status.toUpperCase()) {
            case 'MATCH':
                return 'status-match';
            case 'MISMATCH':
                return 'status-mismatch';
            case 'MISSING':
                return 'status-missing';
            default:
                return 'status-mismatch';
        }
    }

    function setLoadingState(loading) {
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
        console.log(`Toast: ${type} - ${message}`);
        
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
            color: 'black',
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

    // Global functions for action buttons
    window.executeAction = async function(action, inputValue) {
        console.log(`Executing action: ${action} for ${inputValue}`);
        
        try {
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
        }
    };

    window.showDetails = function(inputValue) {
        if (window.lastReconciliationData) {
            console.log('Showing details for:', inputValue);
            console.log('Full data:', window.lastReconciliationData);
            showToast('Details logged to console', 'info');
        }
    };

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

        // Auto-close after 10 seconds
        setTimeout(() => {
            if (document.body.contains(resultModal)) {
                document.body.removeChild(resultModal);
            }
        }, 10000);
    }

    // Add required CSS for new components
    if (!document.querySelector('#dynamic-styles')) {
        const style = document.createElement('style');
        style.id = 'dynamic-styles';
        style.textContent = `
            .validation-error {
                background: var(--error-color);
                color: white;
                padding: 1rem;
                border-radius: var(--radius-lg);
                margin: 1rem 0;
                display: flex;
                align-items: center;
                gap: 0.5rem;
                animation: fadeInUp 0.3s ease-out;
            }
            
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
                margin-bottom: var(--spacing-md);
                color: var(--gray-300);
            }
            
            .scenario-content strong {
                color: var(--white);
            }
            
            .recommended-actions {
                margin-top: var(--spacing-lg);
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
            
            .action-result-modal {
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.5);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 2000;
                animation: fadeIn 0.3s ease-out;
            }
            
            .modal-content {
                background: var(--bg-card);
                color: var(--white);
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
                color: var(--success-color);
                font-size: var(--font-size-xl);
            }
            
            .modal-close {
                background: none;
                border: none;
                font-size: var(--font-size-xl);
                cursor: pointer;
                color: var(--gray-400);
                padding: var(--spacing-xs);
            }
            
            .modal-close:hover {
                color: var(--white);
            }
            
            .modal-body {
                padding: var(--spacing-xl);
            }
            
            .modal-body h4 {
                color: var(--accent-color);
                margin-bottom: var(--spacing-lg);
            }
            
            .modal-body p {
                margin-bottom: var(--spacing-sm);
                color: var(--gray-300);
            }
            
            .modal-body strong {
                color: var(--white);
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @keyframes fadeInUp {
                from { opacity: 0; transform: translateY(20px); }
                to { opacity: 1; transform: translateY(0); }
            }
            
            @keyframes slideInRight {
                from { transform: translateX(100%); opacity: 0; }
                to { transform: translateX(0); opacity: 1; }
            }
            
            @keyframes slideOutRight {
                from { transform: translateX(0); opacity: 1; }
                to { transform: translateX(100%); opacity: 0; }
            }
            
            .results-section.show {
                display: block;
                animation: fadeInUp 0.8s ease-out;
            }
        `;
        document.head.appendChild(style);
    }

    console.log('OSCAR Reconciliation Tool JavaScript loaded successfully');
});
