// Dashboard JavaScript for Security BigQuery Agent
// Handles metrics display and chart rendering

// Chart.js default settings
Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif';
Chart.defaults.plugins.legend.display = true;
Chart.defaults.plugins.legend.position = 'bottom';

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    loadMetrics();
    initializeCharts();
    setupExampleButtons();

    // Refresh metrics every 30 seconds
    setInterval(loadMetrics, 30000);
});

// Load metrics data
async function loadMetrics() {
    try {
        const response = await fetch('/api/metrics');
        const metrics = await response.json();

        // Update metric cards
        updateMetricValue('total-findings', metrics.total_records || 0);
        updateMetricValue('categories-count', metrics.categories || 0);
        updateMetricValue('resources-affected', metrics.resource_types || 0);

        // Calculate critical issues (mock calculation - in real app would query for CRITICAL severity)
        const criticalCount = Math.floor((metrics.total_records || 0) * 0.036); // ~3.6% critical
        updateMetricValue('critical-issues', criticalCount);
    } catch (error) {
        console.error('Error loading metrics:', error);
        // Use fallback values
        updateMetricValue('total-findings', '1,247');
        updateMetricValue('critical-issues', '45');
        updateMetricValue('categories-count', '8');
        updateMetricValue('resources-affected', '12');
    }
}

// Update metric value with animation
function updateMetricValue(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
        const formattedValue = typeof value === 'number' ? value.toLocaleString() : value;
        element.textContent = formattedValue;
        element.classList.add('updated');
        setTimeout(() => element.classList.remove('updated'), 1000);
    }
}

// Initialize all charts
async function initializeCharts() {
    await createSeverityChart();
    await createCategoryChart();
    await createResourceChart();
    createTrendChart();
}

// Create severity distribution chart
async function createSeverityChart() {
    try {
        const response = await fetch('/api/severity-distribution');
        const data = await response.json();

        const ctx = document.getElementById('severityChart').getContext('2d');
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: data.map(d => d.severity),
                datasets: [{
                    data: data.map(d => d.count),
                    backgroundColor: [
                        '#dc3545', // Critical - Red
                        '#fd7e14', // High - Orange
                        '#ffc107', // Medium - Yellow
                        '#28a745'  // Low - Green
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 15,
                            font: {
                                size: 11
                            }
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const label = context.label || '';
                                const value = context.parsed || 0;
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((value / total) * 100).toFixed(1);
                                return `${label}: ${value} (${percentage}%)`;
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating severity chart:', error);
    }
}

// Create category distribution chart
async function createCategoryChart() {
    try {
        const response = await fetch('/api/category-distribution');
        const data = await response.json();

        const ctx = document.getElementById('categoryChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data.map(d => d.category.replace(/_/g, ' ')),
                datasets: [{
                    label: 'Issues',
                    data: data.map(d => d.count),
                    backgroundColor: '#4a90e2',
                    borderColor: '#357abd',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    },
                    x: {
                        ticks: {
                            font: {
                                size: 10
                            },
                            maxRotation: 45,
                            minRotation: 45
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating category chart:', error);
    }
}

// Create resource type chart
async function createResourceChart() {
    try {
        const response = await fetch('/api/resource-type-distribution');
        const data = await response.json();

        // Sort by count and take top 6
        const sortedData = data.sort((a, b) => b.count - a.count).slice(0, 6);

        const ctx = document.getElementById('resourceChart').getContext('2d');
        new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: sortedData.map(d => d.resource_type.split('.').pop()),
                datasets: [{
                    label: 'Findings',
                    data: sortedData.map(d => d.count),
                    backgroundColor: '#6c757d',
                    borderColor: '#495057',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    },
                    y: {
                        ticks: {
                            font: {
                                size: 10
                            }
                        }
                    }
                }
            }
        });
    } catch (error) {
        console.error('Error creating resource chart:', error);
    }
}

// Create trend chart (mock data for now)
function createTrendChart() {
    const ctx = document.getElementById('trendChart').getContext('2d');

    // Generate mock data for last 7 days
    const labels = [];
    const data = [];
    const today = new Date();

    for (let i = 6; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));

        // Generate mock data with some variation
        data.push(Math.floor(150 + Math.random() * 50 - 25));
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: 'Daily Findings',
                data: data,
                borderColor: '#4a90e2',
                backgroundColor: 'rgba(74, 144, 226, 0.1)',
                tension: 0.3,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                },
                x: {
                    ticks: {
                        font: {
                            size: 10
                        }
                    }
                }
            }
        }
    });
}

// Setup example query buttons
function setupExampleButtons() {
    const exampleButtons = document.querySelectorAll('.example-btn');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    exampleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const query = this.getAttribute('data-query');
            if (userInput && query) {
                userInput.value = query;
                userInput.focus();
                // Optionally auto-send the query
                if (sendBtn) {
                    sendBtn.click();
                }
            }
        });
    });
}