/**
 * Student Performance Prediction Dashboard
 * Module: static/js/dashboard.js
 *
 * Handles API interaction, model status checks, metrics rendering,
 * Chart.js comparison visualizations, prediction requests, and session history.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State (in-memory, current session only)
    const state = {
        metrics: null,
        recentPredictions: [],
        comparisonChart: null,
    };

    // DOM Elements
    const elements = {
        // Status & Health
        systemStatusBadge: document.getElementById('systemStatusBadge'),
        statusText: document.getElementById('statusText'),
        footerApiStatus: document.getElementById('footerApiStatus'),

        // Overview / Metrics
        metricsLoading: document.getElementById('metricsLoading'),
        metricsUnavailable: document.getElementById('metricsUnavailable'),
        overviewMetricsGrid: document.getElementById('overviewMetricsGrid'),
        lrMae: document.getElementById('lrMae'),
        svrMae: document.getElementById('svrMae'),
        lrRmse: document.getElementById('lrRmse'),
        svrRmse: document.getElementById('svrRmse'),
        lrR2: document.getElementById('lrR2'),
        svrR2: document.getElementById('svrR2'),

        // Prediction Form
        form: document.getElementById('predictionForm'),
        submitBtn: document.getElementById('submitBtn'),
        btnText: document.getElementById('btnText'),
        btnSpinner: document.getElementById('btnSpinner'),
        resetBtn: document.getElementById('resetBtn'),
        formAlert: document.getElementById('formAlert'),

        // Inputs
        g1Input: document.getElementById('g1Input'),
        g2Input: document.getElementById('g2Input'),
        studytimeInput: document.getElementById('studytimeInput'),
        failuresInput: document.getElementById('failuresInput'),
        absencesInput: document.getElementById('absencesInput'),

        // Field Errors
        g1Error: document.getElementById('g1Error'),
        g2Error: document.getElementById('g2Error'),
        studytimeError: document.getElementById('studytimeError'),
        failuresError: document.getElementById('failuresError'),
        absencesError: document.getElementById('absencesError'),

        // Results Card
        resultPlaceholder: document.getElementById('resultPlaceholder'),
        resultContent: document.getElementById('resultContent'),
        resultTimestamp: document.getElementById('resultTimestamp'),
        lrPredScore: document.getElementById('lrPredScore'),
        svrPredScore: document.getElementById('svrPredScore'),
        lrTierTag: document.getElementById('lrTierTag'),
        svrTierTag: document.getElementById('svrTierTag'),
        modelDelta: document.getElementById('modelDelta'),

        // Chart & Fallback
        chartCanvas: document.getElementById('modelComparisonChart'),
        chartFallbackMessage: document.getElementById('chartFallbackMessage'),

        // Recent Predictions
        sessionTableBody: document.getElementById('sessionTableBody'),
        clearHistoryBtn: document.getElementById('clearHistoryBtn'),

        // Navigation & Mobile
        mobileMenuBtn: document.getElementById('mobileMenuBtn'),
        appSidebar: document.getElementById('appSidebar'),
        navItems: document.querySelectorAll('.nav-item'),
    };

    /* -------------------------------------------------------------------------
       1. Academic Performance Tier Helper
       ------------------------------------------------------------------------- */
    function getPerformanceTier(score) {
        if (score >= 16.0) return { label: 'Distinction', className: 'tier-distinction' };
        if (score >= 14.0) return { label: 'Merit', className: 'tier-merit' };
        if (score >= 10.0) return { label: 'Pass', className: 'tier-pass' };
        return { label: 'Fail', className: 'tier-fail' };
    }

    /* -------------------------------------------------------------------------
       2. Health Status Check (/api/health)
       ------------------------------------------------------------------------- */
    async function checkSystemHealth() {
        try {
            const response = await fetch('/api/health');
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            const data = await response.json();

            elements.systemStatusBadge.className = 'status-badge';

            if (data.status === 'healthy') {
                elements.systemStatusBadge.classList.add('status-healthy');
                elements.statusText.textContent = 'Backend Online • Models Loaded';
                elements.footerApiStatus.textContent = 'Online (Models Active)';
            } else {
                elements.systemStatusBadge.classList.add('status-degraded');
                elements.statusText.textContent = 'Backend Online • Models Not Trained';
                elements.footerApiStatus.textContent = 'Degraded (Train models via train_model.py)';
            }
        } catch (error) {
            elements.systemStatusBadge.className = 'status-badge status-offline';
            elements.statusText.textContent = 'Backend Offline';
            elements.footerApiStatus.textContent = 'Offline (Check Flask server)';
        }
    }

    /* -------------------------------------------------------------------------
       3. Fetch & Render Model Metrics (/api/metrics)
       ------------------------------------------------------------------------- */
    async function loadModelMetrics() {
        try {
            elements.metricsLoading.style.display = 'flex';
            elements.metricsUnavailable.style.display = 'none';
            elements.overviewMetricsGrid.style.display = 'none';

            const response = await fetch('/api/metrics');

            if (response.status === 503) {
                // Models/metrics not yet available
                elements.metricsLoading.style.display = 'none';
                elements.metricsUnavailable.style.display = 'flex';
                elements.chartFallbackMessage.style.display = 'flex';
                return;
            }

            if (!response.ok) throw new Error(`HTTP ${response.status}`);

            const data = await response.json();
            state.metrics = data;

            const lr = data.models?.linear_regression?.metrics;
            const svr = data.models?.svr?.metrics;

            if (lr && svr) {
                elements.lrMae.textContent = lr.mae.toFixed(3);
                elements.svrMae.textContent = svr.mae.toFixed(3);

                elements.lrRmse.textContent = lr.rmse.toFixed(3);
                elements.svrRmse.textContent = svr.rmse.toFixed(3);

                elements.lrR2.textContent = lr.r2.toFixed(3);
                elements.svrR2.textContent = svr.r2.toFixed(3);

                elements.metricsLoading.style.display = 'none';
                elements.overviewMetricsGrid.style.display = 'grid';

                renderComparisonChart(lr, svr);
            } else {
                elements.metricsLoading.style.display = 'none';
                elements.metricsUnavailable.style.display = 'flex';
            }
        } catch (err) {
            elements.metricsLoading.style.display = 'none';
            elements.metricsUnavailable.style.display = 'flex';
            elements.chartFallbackMessage.style.display = 'flex';
        }
    }

    /* -------------------------------------------------------------------------
       4. Render Model Comparison Chart (Chart.js)
       ------------------------------------------------------------------------- */
    function renderComparisonChart(lr, svr) {
        if (!elements.chartCanvas || typeof Chart === 'undefined') {
            if (elements.chartFallbackMessage) {
                elements.chartFallbackMessage.style.display = 'flex';
                elements.chartFallbackMessage.innerHTML = '<p>Chart.js library is loading or unavailable. Metric cards above show verified evaluation values.</p>';
            }
            return;
        }

        elements.chartFallbackMessage.style.display = 'none';

        if (state.comparisonChart) {
            state.comparisonChart.destroy();
        }

        const ctx = elements.chartCanvas.getContext('2d');

        state.comparisonChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MAE (Lower is better)', 'RMSE (Lower is better)', 'R² Score (Higher is better)'],
                datasets: [
                    {
                        label: 'Linear Regression',
                        data: [lr.mae, lr.rmse, lr.r2],
                        backgroundColor: '#3b82f6',
                        borderColor: '#1d4ed8',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                    {
                        label: 'Support Vector Regression (SVR)',
                        data: [svr.mae, svr.rmse, svr.r2],
                        backgroundColor: '#8b5cf6',
                        borderColor: '#6d28d9',
                        borderWidth: 1,
                        borderRadius: 4,
                    },
                ],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            font: { family: '-apple-system, BlinkMacSystemFont, Segoe UI, Roboto', size: 12 },
                            boxWidth: 14,
                        },
                    },
                    tooltip: {
                        backgroundColor: '#0f172a',
                        titleFont: { size: 13, weight: '600' },
                        bodyFont: { size: 12 },
                        padding: 10,
                        callbacks: {
                            label: (context) => ` ${context.dataset.label}: ${context.raw.toFixed(4)}`,
                        },
                    },
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: { font: { size: 12, weight: '500' } },
                    },
                    y: {
                        beginAtZero: true,
                        grid: { color: '#e2e8f0' },
                        ticks: { font: { size: 11 } },
                    },
                },
            },
        });
    }

    /* -------------------------------------------------------------------------
       5. Client-Side Input Validation
       ------------------------------------------------------------------------- */
    function clearFormErrors() {
        elements.formAlert.style.display = 'none';
        elements.formAlert.textContent = '';
        const errorSpans = [elements.g1Error, elements.g2Error, elements.studytimeError, elements.failuresError, elements.absencesError];
        errorSpans.forEach((span) => {
            span.style.display = 'none';
            span.textContent = '';
        });
    }

    function validateInputs() {
        clearFormErrors();
        let isValid = true;

        const g1 = parseFloat(elements.g1Input.value);
        if (isNaN(g1) || g1 < 0 || g1 > 20) {
            elements.g1Error.textContent = 'G1 must be between 0.0 and 20.0 points.';
            elements.g1Error.style.display = 'block';
            isValid = false;
        }

        const g2 = parseFloat(elements.g2Input.value);
        if (isNaN(g2) || g2 < 0 || g2 > 20) {
            elements.g2Error.textContent = 'G2 must be between 0.0 and 20.0 points.';
            elements.g2Error.style.display = 'block';
            isValid = false;
        }

        const studytime = parseInt(elements.studytimeInput.value, 10);
        if (isNaN(studytime) || studytime < 1 || studytime > 4) {
            elements.studytimeError.textContent = 'Select a study time category (1 to 4).';
            elements.studytimeError.style.display = 'block';
            isValid = false;
        }

        const failures = parseInt(elements.failuresInput.value, 10);
        if (isNaN(failures) || failures < 0 || failures > 4) {
            elements.failuresError.textContent = 'Select past failures (0 to 4).';
            elements.failuresError.style.display = 'block';
            isValid = false;
        }

        const absences = parseInt(elements.absencesInput.value, 10);
        if (isNaN(absences) || absences < 0 || absences > 100) {
            elements.absencesError.textContent = 'Absences must be between 0 and 100.';
            elements.absencesError.style.display = 'block';
            isValid = false;
        }

        return isValid ? { G1: g1, G2: g2, studytime: studytime, failures: failures, absences: absences } : null;
    }

    /* -------------------------------------------------------------------------
       6. Form Submission & Prediction Handler (POST /api/predict)
       ------------------------------------------------------------------------- */
    async function handlePredictionSubmit(e) {
        e.preventDefault();
        const payload = validateInputs();
        if (!payload) return;

        // UI loading state
        elements.submitBtn.disabled = true;
        elements.btnText.textContent = 'Computing Scores...';
        elements.btnSpinner.style.display = 'inline-block';

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || `Prediction failed with status ${response.status}`);
            }

            // Extract results
            const lrPred = data.predictions.linear_regression;
            const svrPred = data.predictions.svr;
            const lrTier = getPerformanceTier(lrPred);
            const svrTier = getPerformanceTier(svrPred);

            // Display in results card
            elements.lrPredScore.textContent = lrPred.toFixed(2);
            elements.svrPredScore.textContent = svrPred.toFixed(2);

            elements.lrTierTag.textContent = lrTier.label;
            elements.svrTierTag.textContent = svrTier.label;

            const delta = Math.abs(lrPred - svrPred);
            elements.modelDelta.textContent = `${delta.toFixed(2)} points`;

            elements.resultTimestamp.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

            elements.resultPlaceholder.style.display = 'none';
            elements.resultContent.style.display = 'flex';

            // Add to session history
            state.recentPredictions.unshift({
                id: state.recentPredictions.length + 1,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                g1: payload.G1,
                g2: payload.G2,
                studytime: payload.studytime,
                failures: payload.failures,
                absences: payload.absences,
                lr: lrPred.toFixed(2),
                svr: svrPred.toFixed(2),
            });

            renderRecentPredictions();
        } catch (error) {
            elements.formAlert.textContent = error.message;
            elements.formAlert.style.display = 'block';
        } finally {
            elements.submitBtn.disabled = false;
            elements.btnText.textContent = 'Generate Prediction';
            elements.btnSpinner.style.display = 'none';
        }
    }

    /* -------------------------------------------------------------------------
       7. Session Recent Predictions Table
       ------------------------------------------------------------------------- */
    function renderRecentPredictions() {
        if (!elements.sessionTableBody) return;

        if (state.recentPredictions.length === 0) {
            elements.sessionTableBody.innerHTML = `
                <tr id="emptySessionRow">
                    <td colspan="9" class="empty-table-cell">No predictions generated yet in this session.</td>
                </tr>
            `;
            elements.clearHistoryBtn.disabled = true;
            return;
        }

        elements.clearHistoryBtn.disabled = false;
        elements.sessionTableBody.innerHTML = state.recentPredictions
            .map(
                (item) => `
                <tr>
                    <td><strong>#${item.id}</strong></td>
                    <td>${item.time}</td>
                    <td>${item.g1}</td>
                    <td>${item.g2}</td>
                    <td>Scale ${item.studytime}</td>
                    <td>${item.failures}</td>
                    <td>${item.absences}</td>
                    <td><strong>${item.lr}</strong> / 20</td>
                    <td><strong>${item.svr}</strong> / 20</td>
                </tr>
            `
            )
            .join('');
    }

    function clearRecentPredictions() {
        state.recentPredictions = [];
        renderRecentPredictions();
    }

    /* -------------------------------------------------------------------------
       8. Navigation & UI Listeners
       ------------------------------------------------------------------------- */
    // Reset Form
    elements.resetBtn.addEventListener('click', () => {
        elements.form.reset();
        clearFormErrors();
        elements.g1Input.value = '12';
        elements.g2Input.value = '13';
        elements.studytimeInput.value = '2';
        elements.failuresInput.value = '0';
        elements.absencesInput.value = '4';
    });

    // Form Submit
    elements.form.addEventListener('submit', handlePredictionSubmit);

    // Clear History Button
    elements.clearHistoryBtn.addEventListener('click', clearRecentPredictions);

    // Mobile Sidebar Toggle
    if (elements.mobileMenuBtn && elements.appSidebar) {
        elements.mobileMenuBtn.addEventListener('click', () => {
            elements.appSidebar.classList.toggle('open');
        });

        // Close sidebar when clicking outside on mobile
        document.addEventListener('click', (e) => {
            if (
                elements.appSidebar.classList.contains('open') &&
                !elements.appSidebar.contains(e.target) &&
                !elements.mobileMenuBtn.contains(e.target)
            ) {
                elements.appSidebar.classList.remove('open');
            }
        });
    }

    // Active Navigation Highlight
    elements.navItems.forEach((item) => {
        item.addEventListener('click', () => {
            elements.navItems.forEach((nav) => nav.classList.remove('active'));
            item.classList.add('active');
            if (window.innerWidth <= 900 && elements.appSidebar) {
                elements.appSidebar.classList.remove('open');
            }
        });
    });

    /* -------------------------------------------------------------------------
       9. Initialize on Page Load
       ------------------------------------------------------------------------- */
    checkSystemHealth();
    loadModelMetrics();
    renderRecentPredictions();

    // Periodic health check every 30 seconds
    setInterval(checkSystemHealth, 30000);
});
