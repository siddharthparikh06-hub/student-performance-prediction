/**
 * Student Performance & College CGPA Prediction Dashboard
 * Module: static/js/dashboard.js
 *
 * Handles domain switching (College CGPA vs School G3),
 * API requests, input validation, Chart.js benchmarks, and session history.
 */

document.addEventListener('DOMContentLoaded', () => {
    // Application State (in-memory, current session only)
    const state = {
        currentDomain: 'college', // 'college' or 'school'
        collegeMetrics: null,
        schoolMetrics: null,
        recentCollegePredictions: [],
        recentSchoolPredictions: [],
        collegeChart: null,
        schoolChart: null,
    };

    // DOM Elements
    const elements = {
        // Status & Topbar
        topbarTitle: document.getElementById('topbarTitle'),
        topbarSubtitle: document.getElementById('topbarSubtitle'),
        systemStatusBadge: document.getElementById('systemStatusBadge'),
        statusText: document.getElementById('statusText'),
        footerApiStatus: document.getElementById('footerApiStatus'),

        // Domain Switchers
        sideBtnCollege: document.getElementById('sideBtnCollege'),
        sideBtnSchool: document.getElementById('sideBtnSchool'),
        topBtnCollege: document.getElementById('topBtnCollege'),
        topBtnSchool: document.getElementById('topBtnSchool'),
        collegeView: document.getElementById('collegeView'),
        schoolView: document.getElementById('schoolView'),
        sidebarSubtext: document.getElementById('sidebarSubtext'),
        sidebarSplitTag: document.getElementById('sidebarSplitTag'),

        // College Overview / Metrics
        collegeMetricsLoading: document.getElementById('collegeMetricsLoading'),
        collegeMetricsUnavailable: document.getElementById('collegeMetricsUnavailable'),
        collegeMetricsGrid: document.getElementById('collegeMetricsGrid'),
        collegeLrMae: document.getElementById('collegeLrMae'),
        collegeSvrMae: document.getElementById('collegeSvrMae'),
        collegeLrRmse: document.getElementById('collegeLrRmse'),
        collegeSvrRmse: document.getElementById('collegeSvrRmse'),
        collegeLrR2: document.getElementById('collegeLrR2'),
        collegeSvrR2: document.getElementById('collegeSvrR2'),

        // College Prediction Form
        collegeForm: document.getElementById('collegePredictionForm'),
        collegeStudyHoursInput: document.getElementById('collegeStudyHoursInput'),
        collegeStudyHoursError: document.getElementById('collegeStudyHoursError'),
        collegeStudentName: document.getElementById('collegeStudentName'),
        collegeStudentId: document.getElementById('collegeStudentId'),
        collegeDepartment: document.getElementById('collegeDepartment'),
        collegeSemester: document.getElementById('collegeSemester'),
        collegeSubmitBtn: document.getElementById('collegeSubmitBtn'),
        collegeBtnText: document.getElementById('collegeBtnText'),
        collegeBtnSpinner: document.getElementById('collegeBtnSpinner'),
        collegeResetBtn: document.getElementById('collegeResetBtn'),
        collegeFormAlert: document.getElementById('collegeFormAlert'),

        // College Results
        collegeResultPlaceholder: document.getElementById('collegeResultPlaceholder'),
        collegeResultContent: document.getElementById('collegeResultContent'),
        collegeResultTimestamp: document.getElementById('collegeResultTimestamp'),
        collegeMetadataEcho: document.getElementById('collegeMetadataEcho'),
        collegeMetadataEchoText: document.getElementById('collegeMetadataEchoText'),
        collegeLrScore: document.getElementById('collegeLrScore'),
        collegeSvrScore: document.getElementById('collegeSvrScore'),
        collegeLrTierTag: document.getElementById('collegeLrTierTag'),
        collegeSvrTierTag: document.getElementById('collegeSvrTierTag'),
        collegeModelDelta: document.getElementById('collegeModelDelta'),

        // College Chart & Recent
        collegeChartCanvas: document.getElementById('collegeComparisonChart'),
        collegeChartFallback: document.getElementById('collegeChartFallback'),
        collegeSessionTableBody: document.getElementById('collegeSessionTableBody'),
        clearCollegeHistoryBtn: document.getElementById('clearCollegeHistoryBtn'),

        // School Elements
        metricsLoading: document.getElementById('metricsLoading'),
        metricsUnavailable: document.getElementById('metricsUnavailable'),
        overviewMetricsGrid: document.getElementById('overviewMetricsGrid'),
        lrMae: document.getElementById('lrMae'),
        svrMae: document.getElementById('svrMae'),
        lrRmse: document.getElementById('lrRmse'),
        svrRmse: document.getElementById('svrRmse'),
        lrR2: document.getElementById('lrR2'),
        svrR2: document.getElementById('svrR2'),
        form: document.getElementById('predictionForm'),
        submitBtn: document.getElementById('submitBtn'),
        btnText: document.getElementById('btnText'),
        btnSpinner: document.getElementById('btnSpinner'),
        resetBtn: document.getElementById('resetBtn'),
        formAlert: document.getElementById('formAlert'),
        g1Input: document.getElementById('g1Input'),
        g2Input: document.getElementById('g2Input'),
        studytimeInput: document.getElementById('studytimeInput'),
        failuresInput: document.getElementById('failuresInput'),
        absencesInput: document.getElementById('absencesInput'),
        g1Error: document.getElementById('g1Error'),
        g2Error: document.getElementById('g2Error'),
        studytimeError: document.getElementById('studytimeError'),
        failuresError: document.getElementById('failuresError'),
        absencesError: document.getElementById('absencesError'),
        resultPlaceholder: document.getElementById('resultPlaceholder'),
        resultContent: document.getElementById('resultContent'),
        resultTimestamp: document.getElementById('resultTimestamp'),
        lrPredScore: document.getElementById('lrPredScore'),
        svrPredScore: document.getElementById('svrPredScore'),
        lrTierTag: document.getElementById('lrTierTag'),
        svrTierTag: document.getElementById('svrTierTag'),
        modelDelta: document.getElementById('modelDelta'),
        chartCanvas: document.getElementById('modelComparisonChart'),
        chartFallbackMessage: document.getElementById('chartFallbackMessage'),
        sessionTableBody: document.getElementById('sessionTableBody'),
        clearHistoryBtn: document.getElementById('clearHistoryBtn'),

        // Navigation
        mobileMenuBtn: document.getElementById('mobileMenuBtn'),
        appSidebar: document.getElementById('appSidebar'),
        navItems: document.querySelectorAll('.nav-item'),
    };

    /* -------------------------------------------------------------------------
       1. Academic Performance Tiers
       ------------------------------------------------------------------------- */
    function getCollegeCgpaTier(cgpa) {
        if (cgpa >= 9.0) return { label: 'Outstanding', className: 'tier-distinction' };
        if (cgpa >= 8.0) return { label: 'Excellent', className: 'tier-merit' };
        if (cgpa >= 7.0) return { label: 'Very Good', className: 'tier-pass' };
        if (cgpa >= 6.0) return { label: 'Good', className: 'tier-pass' };
        if (cgpa >= 4.0) return { label: 'Pass', className: 'tier-pass' };
        return { label: 'Remedial', className: 'tier-fail' };
    }

    function getSchoolGradeTier(score) {
        if (score >= 16.0) return { label: 'Distinction', className: 'tier-distinction' };
        if (score >= 14.0) return { label: 'Merit', className: 'tier-merit' };
        if (score >= 10.0) return { label: 'Pass', className: 'tier-pass' };
        return { label: 'Fail', className: 'tier-fail' };
    }

    /* -------------------------------------------------------------------------
       2. Domain Switcher Logic
       ------------------------------------------------------------------------- */
    function switchDomain(domain) {
        state.currentDomain = domain;

        if (domain === 'college') {
            elements.collegeView.style.display = 'flex';
            elements.schoolView.style.display = 'none';

            elements.sideBtnCollege.classList.add('active');
            elements.sideBtnSchool.classList.remove('active');
            elements.topBtnCollege.classList.add('active');
            elements.topBtnSchool.classList.remove('active');

            elements.topbarTitle.textContent = 'College CGPA Prediction System';
            elements.topbarSubtitle.textContent = 'Machine Learning Regression Dashboard • Study Hours to CGPA';
            elements.sidebarSplitTag.textContent = 'Active: College CGPA (0.0-10.0)';

            checkCollegeHealth();
            if (!state.collegeMetrics) loadCollegeMetrics();
        } else {
            elements.collegeView.style.display = 'none';
            elements.schoolView.style.display = 'flex';

            elements.sideBtnCollege.classList.remove('active');
            elements.sideBtnSchool.classList.add('active');
            elements.topBtnCollege.classList.remove('active');
            elements.topBtnSchool.classList.add('active');

            elements.topbarTitle.textContent = 'School Performance Prediction System';
            elements.topbarSubtitle.textContent = 'Machine Learning Regression Dashboard • UCI Student Dataset Standard';
            elements.sidebarSplitTag.textContent = 'Active: School G3 (0.0-20.0)';

            checkSchoolHealth();
            if (!state.schoolMetrics) loadSchoolMetrics();
        }
    }

    elements.sideBtnCollege.addEventListener('click', () => switchDomain('college'));
    elements.topBtnCollege.addEventListener('click', () => switchDomain('college'));
    elements.sideBtnSchool.addEventListener('click', () => switchDomain('school'));
    elements.topBtnSchool.addEventListener('click', () => switchDomain('school'));

    /* -------------------------------------------------------------------------
       3. Health Status Checks
       ------------------------------------------------------------------------- */
    async function checkCollegeHealth() {
        try {
            const resp = await fetch('/api/college/health');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            elements.systemStatusBadge.className = 'status-badge';
            if (data.status === 'healthy') {
                elements.systemStatusBadge.classList.add('status-healthy');
                elements.statusText.textContent = 'Backend Online • College Models Active';
                elements.footerApiStatus.textContent = 'Online (College Models Ready)';
            } else {
                elements.systemStatusBadge.classList.add('status-degraded');
                elements.statusText.textContent = 'Backend Online • College Models Not Trained';
                elements.footerApiStatus.textContent = 'Degraded (Run train_college_model.py)';
            }
        } catch (e) {
            elements.systemStatusBadge.className = 'status-badge status-offline';
            elements.statusText.textContent = 'Backend Offline';
            elements.footerApiStatus.textContent = 'Offline (Check Flask server)';
        }
    }

    async function checkSchoolHealth() {
        try {
            const resp = await fetch('/api/health');
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();

            elements.systemStatusBadge.className = 'status-badge';
            if (data.status === 'healthy') {
                elements.systemStatusBadge.classList.add('status-healthy');
                elements.statusText.textContent = 'Backend Online • School Models Active';
                elements.footerApiStatus.textContent = 'Online (School Models Ready)';
            } else {
                elements.systemStatusBadge.classList.add('status-degraded');
                elements.statusText.textContent = 'Backend Online • School Models Not Trained';
                elements.footerApiStatus.textContent = 'Degraded (Run train_model.py)';
            }
        } catch (e) {
            elements.systemStatusBadge.className = 'status-badge status-offline';
            elements.statusText.textContent = 'Backend Offline';
            elements.footerApiStatus.textContent = 'Offline (Check Flask server)';
        }
    }

    /* -------------------------------------------------------------------------
       4. College Metrics & Chart (/api/college/metrics)
       ------------------------------------------------------------------------- */
    async function loadCollegeMetrics() {
        try {
            elements.collegeMetricsLoading.style.display = 'flex';
            elements.collegeMetricsUnavailable.style.display = 'none';
            elements.collegeMetricsGrid.style.display = 'none';

            const resp = await fetch('/api/college/metrics');
            if (resp.status === 503) {
                elements.collegeMetricsLoading.style.display = 'none';
                elements.collegeMetricsUnavailable.style.display = 'flex';
                elements.collegeChartFallback.style.display = 'flex';
                return;
            }
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            const data = await resp.json();
            state.collegeMetrics = data;

            const lr = data.models?.linear_regression?.metrics;
            const svr = data.models?.svr?.metrics;

            if (lr && svr) {
                elements.collegeLrMae.textContent = lr.mae.toFixed(4);
                elements.collegeSvrMae.textContent = svr.mae.toFixed(4);

                elements.collegeLrRmse.textContent = lr.rmse.toFixed(4);
                elements.collegeSvrRmse.textContent = svr.rmse.toFixed(4);

                elements.collegeLrR2.textContent = lr.r2.toFixed(4);
                elements.collegeSvrR2.textContent = svr.r2.toFixed(4);

                elements.collegeMetricsLoading.style.display = 'none';
                elements.collegeMetricsGrid.style.display = 'grid';

                renderCollegeChart(lr, svr);
            }
        } catch (e) {
            elements.collegeMetricsLoading.style.display = 'none';
            elements.collegeMetricsUnavailable.style.display = 'flex';
            elements.collegeChartFallback.style.display = 'flex';
        }
    }

    function renderCollegeChart(lr, svr) {
        if (!elements.collegeChartCanvas || typeof Chart === 'undefined') {
            if (elements.collegeChartFallback) elements.collegeChartFallback.style.display = 'flex';
            return;
        }

        elements.collegeChartFallback.style.display = 'none';
        if (state.collegeChart) state.collegeChart.destroy();

        const ctx = elements.collegeChartCanvas.getContext('2d');
        state.collegeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['MAE (Lower is better)', 'RMSE (Lower is better)', 'R² Score (Higher is better)'],
                datasets: [
                    {
                        label: 'Linear Regression',
                        data: [lr.mae, lr.rmse, lr.r2],
                        backgroundColor: '#0284c7',
                        borderColor: '#0369a1',
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
                    legend: { position: 'top' },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => ` ${ctx.dataset.label}: ${ctx.raw.toFixed(4)}`,
                        },
                    },
                },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, grid: { color: '#e2e8f0' } },
                },
            },
        });
    }

    /* -------------------------------------------------------------------------
       5. College Prediction Handler (POST /api/college/predict)
       ------------------------------------------------------------------------- */
    elements.collegeForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        elements.collegeFormAlert.style.display = 'none';
        elements.collegeStudyHoursError.style.display = 'none';

        const studyHoursVal = parseFloat(elements.collegeStudyHoursInput.value);
        if (isNaN(studyHoursVal) || studyHoursVal < 0 || studyHoursVal > 24) {
            elements.collegeStudyHoursError.textContent = 'Daily study hours must be a number between 0.0 and 24.0.';
            elements.collegeStudyHoursError.style.display = 'block';
            return;
        }

        const payload = {
            Study_Hours: studyHoursVal,
            student_name: elements.collegeStudentName.value.trim() || undefined,
            student_id: elements.collegeStudentId.value.trim() || undefined,
            department: elements.collegeDepartment.value.trim() || undefined,
            semester: elements.collegeSemester.value || undefined,
        };

        // Loading state
        elements.collegeSubmitBtn.disabled = true;
        elements.collegeBtnText.textContent = 'Predicting CGPA...';
        elements.collegeBtnSpinner.style.display = 'inline-block';

        try {
            const resp = await fetch('/api/college/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || `Prediction failed (${resp.status})`);

            const lrPred = data.predictions.linear_regression;
            const svrPred = data.predictions.svr;
            const lrTier = getCollegeCgpaTier(lrPred);
            const svrTier = getCollegeCgpaTier(svrPred);

            elements.collegeLrScore.textContent = lrPred.toFixed(2);
            elements.collegeSvrScore.textContent = svrPred.toFixed(2);
            elements.collegeLrTierTag.textContent = lrTier.label;
            elements.collegeSvrTierTag.textContent = svrTier.label;

            const delta = Math.abs(lrPred - svrPred);
            elements.collegeModelDelta.textContent = `${delta.toFixed(2)} CGPA points`;

            elements.collegeResultTimestamp.textContent = new Date().toLocaleTimeString();

            // Metadata Echo
            const metaParts = [];
            if (data.metadata?.student_name) metaParts.push(`Student: <strong>${data.metadata.student_name}</strong>`);
            if (data.metadata?.student_id) metaParts.push(`ID: <strong>${data.metadata.student_id}</strong>`);
            if (data.metadata?.department) metaParts.push(`Dept: <strong>${data.metadata.department}</strong>`);
            if (data.metadata?.semester) metaParts.push(`Semester: <strong>${data.metadata.semester}</strong>`);

            if (metaParts.length > 0) {
                elements.collegeMetadataEchoText.innerHTML = metaParts.join(' • ');
                elements.collegeMetadataEcho.style.display = 'block';
            } else {
                elements.collegeMetadataEcho.style.display = 'none';
            }

            elements.collegeResultPlaceholder.style.display = 'none';
            elements.collegeResultContent.style.display = 'flex';

            // Add to session history
            state.recentCollegePredictions.unshift({
                id: state.recentCollegePredictions.length + 1,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                studentInfo: (data.metadata?.student_name || data.metadata?.student_id)
                    ? `${data.metadata.student_name || 'Student'} (${data.metadata.student_id || '-'})`
                    : 'Anonymous',
                studyHours: studyHoursVal.toFixed(1),
                lr: lrPred.toFixed(2),
                svr: svrPred.toFixed(2),
                deptSem: (data.metadata?.department || data.metadata?.semester)
                    ? `${data.metadata.department || '-'} / ${data.metadata.semester || '-'}`
                    : '-',
            });

            renderCollegeRecentPredictions();
        } catch (err) {
            elements.collegeFormAlert.textContent = err.message;
            elements.collegeFormAlert.style.display = 'block';
        } finally {
            elements.collegeSubmitBtn.disabled = false;
            elements.collegeBtnText.textContent = 'Compute Predicted CGPA';
            elements.collegeBtnSpinner.style.display = 'none';
        }
    });

    elements.collegeResetBtn.addEventListener('click', () => {
        elements.collegeForm.reset();
        elements.collegeStudyHoursInput.value = '6.5';
        elements.collegeFormAlert.style.display = 'none';
        elements.collegeStudyHoursError.style.display = 'none';
    });

    function renderCollegeRecentPredictions() {
        if (!elements.collegeSessionTableBody) return;

        if (state.recentCollegePredictions.length === 0) {
            elements.collegeSessionTableBody.innerHTML = `
                <tr id="emptyCollegeSessionRow">
                    <td colspan="7" class="empty-table-cell">No college predictions generated yet in this session.</td>
                </tr>
            `;
            elements.clearCollegeHistoryBtn.disabled = true;
            return;
        }

        elements.clearCollegeHistoryBtn.disabled = false;
        elements.collegeSessionTableBody.innerHTML = state.recentCollegePredictions
            .map(
                (item) => `
                <tr>
                    <td><strong>#${item.id}</strong></td>
                    <td>${item.time}</td>
                    <td>${item.studentInfo}</td>
                    <td><strong>${item.studyHours} hrs/day</strong></td>
                    <td><strong>${item.lr}</strong> / 10.0</td>
                    <td><strong>${item.svr}</strong> / 10.0</td>
                    <td>${item.deptSem}</td>
                </tr>
            `
            )
            .join('');
    }

    elements.clearCollegeHistoryBtn.addEventListener('click', () => {
        state.recentCollegePredictions = [];
        renderCollegeRecentPredictions();
    });

    /* -------------------------------------------------------------------------
       6. School Prediction Handler (POST /api/predict)
       ------------------------------------------------------------------------- */
    async function loadSchoolMetrics() {
        try {
            elements.metricsLoading.style.display = 'flex';
            elements.metricsUnavailable.style.display = 'none';
            elements.overviewMetricsGrid.style.display = 'none';

            const resp = await fetch('/api/metrics');
            if (resp.status === 503) {
                elements.metricsLoading.style.display = 'none';
                elements.metricsUnavailable.style.display = 'flex';
                elements.chartFallbackMessage.style.display = 'flex';
                return;
            }
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

            const data = await resp.json();
            state.schoolMetrics = data;

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

                renderSchoolChart(lr, svr);
            }
        } catch (e) {
            elements.metricsLoading.style.display = 'none';
            elements.metricsUnavailable.style.display = 'flex';
            elements.chartFallbackMessage.style.display = 'flex';
        }
    }

    function renderSchoolChart(lr, svr) {
        if (!elements.chartCanvas || typeof Chart === 'undefined') {
            if (elements.chartFallbackMessage) elements.chartFallbackMessage.style.display = 'flex';
            return;
        }

        elements.chartFallbackMessage.style.display = 'none';
        if (state.schoolChart) state.schoolChart.destroy();

        const ctx = elements.chartCanvas.getContext('2d');
        state.schoolChart = new Chart(ctx, {
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
                plugins: { legend: { position: 'top' } },
                scales: {
                    x: { grid: { display: false } },
                    y: { beginAtZero: true, grid: { color: '#e2e8f0' } },
                },
            },
        });
    }

    elements.form.addEventListener('submit', async (e) => {
        e.preventDefault();
        elements.formAlert.style.display = 'none';

        const g1 = parseFloat(elements.g1Input.value);
        const g2 = parseFloat(elements.g2Input.value);
        const studytime = parseInt(elements.studytimeInput.value, 10);
        const failures = parseInt(elements.failuresInput.value, 10);
        const absences = parseInt(elements.absencesInput.value, 10);

        if (isNaN(g1) || g1 < 0 || g1 > 20 || isNaN(g2) || g2 < 0 || g2 > 20) {
            elements.formAlert.textContent = 'Please enter valid grades between 0 and 20.';
            elements.formAlert.style.display = 'block';
            return;
        }

        const payload = { G1: g1, G2: g2, studytime: studytime, failures: failures, absences: absences };

        elements.submitBtn.disabled = true;
        elements.btnText.textContent = 'Computing Scores...';
        elements.btnSpinner.style.display = 'inline-block';

        try {
            const resp = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || 'Prediction failed');

            const lrPred = data.predictions.linear_regression;
            const svrPred = data.predictions.svr;
            elements.lrPredScore.textContent = lrPred.toFixed(2);
            elements.svrPredScore.textContent = svrPred.toFixed(2);

            const lrTier = getSchoolGradeTier(lrPred);
            const svrTier = getSchoolGradeTier(svrPred);
            elements.lrTierTag.textContent = lrTier.label;
            elements.svrTierTag.textContent = svrTier.label;

            const delta = Math.abs(lrPred - svrPred);
            elements.modelDelta.textContent = `${delta.toFixed(2)} points`;
            elements.resultTimestamp.textContent = new Date().toLocaleTimeString();

            elements.resultPlaceholder.style.display = 'none';
            elements.resultContent.style.display = 'flex';

            state.recentSchoolPredictions.unshift({
                id: state.recentSchoolPredictions.length + 1,
                time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                g1,
                g2,
                studytime,
                failures,
                absences,
                lr: lrPred.toFixed(2),
                svr: svrPred.toFixed(2),
            });
            renderSchoolRecentPredictions();
        } catch (err) {
            elements.formAlert.textContent = err.message;
            elements.formAlert.style.display = 'block';
        } finally {
            elements.submitBtn.disabled = false;
            elements.btnText.textContent = 'Generate Prediction';
            elements.btnSpinner.style.display = 'none';
        }
    });

    elements.resetBtn.addEventListener('click', () => {
        elements.form.reset();
        elements.g1Input.value = '12';
        elements.g2Input.value = '13';
        elements.studytimeInput.value = '2';
        elements.failuresInput.value = '0';
        elements.absencesInput.value = '4';
    });

    function renderSchoolRecentPredictions() {
        if (!elements.sessionTableBody) return;
        if (state.recentSchoolPredictions.length === 0) {
            elements.sessionTableBody.innerHTML = `
                <tr id="emptySessionRow">
                    <td colspan="9" class="empty-table-cell">No school predictions generated yet in this session.</td>
                </tr>
            `;
            elements.clearHistoryBtn.disabled = true;
            return;
        }

        elements.clearHistoryBtn.disabled = false;
        elements.sessionTableBody.innerHTML = state.recentSchoolPredictions
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

    elements.clearHistoryBtn.addEventListener('click', () => {
        state.recentSchoolPredictions = [];
        renderSchoolRecentPredictions();
    });

    /* -------------------------------------------------------------------------
       7. Navigation & Mobile Drawer
       ------------------------------------------------------------------------- */
    if (elements.mobileMenuBtn && elements.appSidebar) {
        elements.mobileMenuBtn.addEventListener('click', () => {
            elements.appSidebar.classList.toggle('open');
        });
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

    elements.navItems.forEach((item) => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            elements.navItems.forEach((nav) => nav.classList.remove('active'));
            item.classList.add('active');

            const targetSection = item.getAttribute('href').replace('#', '');
            let targetEl = null;

            if (state.currentDomain === 'college') {
                targetEl = document.getElementById(`college-${targetSection}`);
            } else {
                targetEl = document.getElementById(targetSection);
            }

            if (targetEl) {
                targetEl.scrollIntoView({ behavior: 'smooth' });
            }

            if (window.innerWidth <= 900 && elements.appSidebar) {
                elements.appSidebar.classList.remove('open');
            }
        });
    });

    /* -------------------------------------------------------------------------
       8. Initial Page Boot
       ------------------------------------------------------------------------- */
    switchDomain('college'); // College CGPA active by default
    renderCollegeRecentPredictions();
    renderSchoolRecentPredictions();

    // Periodic health check
    setInterval(() => {
        if (state.currentDomain === 'college') {
            checkCollegeHealth();
        } else {
            checkSchoolHealth();
        }
    }, 30000);
});
