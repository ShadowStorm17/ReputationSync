// Main JavaScript file for the Reputation System

// Initialize Alpine.js components
document.addEventListener('alpine:init', () => {
    Alpine.data('dashboard', () => ({
        loading: false,
        refreshInterval: null,
        
        init() {
            this.startAutoRefresh();
        },
        
        startAutoRefresh() {
            this.refreshInterval = setInterval(() => {
                this.refreshData();
            }, 30000); // Refresh every 30 seconds
        },
        
        async refreshData() {
            this.loading = true;
            try {
                const response = await fetch('/api/v1/dashboard/refresh');
                const data = await response.json();
                this.updateDashboard(data);
            } catch (error) {
                console.error('Error refreshing dashboard:', error);
            } finally {
                this.loading = false;
            }
        },
        
        updateDashboard(data) {
            // Update stats
            document.getElementById('total-score').textContent = data.total_score;
            document.getElementById('active-users').textContent = data.active_users;
            document.getElementById('sentiment-score').textContent = data.sentiment_score;
            document.getElementById('engagement-rate').textContent = data.engagement_rate;
            
            // Update charts
            this.updateCharts(data);
            
            // Update recent activity
            this.updateRecentActivity(data.recent_activities);
        },
        
        updateCharts(data) {
            // Update reputation trend chart
            const reputationChart = Chart.getChart('reputationChart');
            if (reputationChart) {
                reputationChart.data.labels = data.dates;
                reputationChart.data.datasets[0].data = data.scores;
                reputationChart.update();
            }
            
            // Update sentiment distribution chart
            const sentimentChart = Chart.getChart('sentimentChart');
            if (sentimentChart) {
                sentimentChart.data.datasets[0].data = data.sentiment_distribution;
                sentimentChart.update();
            }
        },
        
        updateRecentActivity(activities) {
            const activityList = document.getElementById('recent-activity-list');
            if (!activityList) return;
            
            activityList.innerHTML = activities.map(activity => `
                <li class="px-4 py-4 sm:px-6">
                    <div class="flex items-center justify-between">
                        <div class="flex items-center">
                            <div class="flex-shrink-0">
                                <img class="h-8 w-8 rounded-full" src="${activity.user_avatar}" alt="">
                            </div>
                            <div class="ml-4">
                                <div class="text-sm font-medium text-gray-900">${activity.user_name}</div>
                                <div class="text-sm text-gray-500">${activity.description}</div>
                            </div>
                        </div>
                        <div class="ml-2 flex-shrink-0 flex">
                            <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${activity.status_class}">
                                ${activity.status}
                            </span>
                        </div>
                    </div>
                </li>
            `).join('');
        }
    }));
});

// Chart.js global configuration
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(0, 0, 0, 0.8)';
Chart.defaults.plugins.tooltip.padding = 12;
Chart.defaults.plugins.tooltip.cornerRadius = 4;
Chart.defaults.plugins.tooltip.titleFont = {
    size: 14,
    weight: 'bold'
};
Chart.defaults.plugins.tooltip.bodyFont = {
    size: 13
};

// Utility functions
function formatNumber(number) {
    return new Intl.NumberFormat().format(number);
}

function formatPercentage(number) {
    return new Intl.NumberFormat('en-US', {
        style: 'percent',
        minimumFractionDigits: 1,
        maximumFractionDigits: 1
    }).format(number / 100);
}

function formatDate(date) {
    return new Intl.DateTimeFormat('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    }).format(new Date(date));
}

// Event listeners
document.addEventListener('DOMContentLoaded', () => {
    // Handle mobile menu toggle
    const menuButton = document.getElementById('mobile-menu-button');
    const mobileMenu = document.getElementById('mobile-menu');
    
    if (menuButton && mobileMenu) {
        menuButton.addEventListener('click', () => {
            mobileMenu.classList.toggle('hidden');
        });
    }
    
    // Handle user menu toggle
    const userMenuButton = document.getElementById('user-menu-button');
    const userMenu = document.getElementById('user-menu');
    
    if (userMenuButton && userMenu) {
        userMenuButton.addEventListener('click', () => {
            userMenu.classList.toggle('hidden');
        });
    }
    
    // Handle dark mode toggle
    const darkModeToggle = document.getElementById('dark-mode-toggle');
    
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', () => {
            document.documentElement.classList.toggle('dark');
            localStorage.setItem('darkMode', document.documentElement.classList.contains('dark'));
        });
        
        // Check for saved dark mode preference
        if (localStorage.getItem('darkMode') === 'true') {
            document.documentElement.classList.add('dark');
        }
    }
}); 