// PhishGuard - Main Application Logic
class PhishGuardApp {
    constructor() {
        this.currentScreen = 'loading';
        this.participantId = this.generateUUID();
        this.sessionId = this.generateUUID();
        this.testEmails = [];
        this.currentEmailIndex = 0;
        this.currentModeIndex = 0;
        this.responses = [];
        this.emailStartTime = null;
        this.timerInterval = null;
        this.currentTimer = 0;

        this.modes = ['human_only', 'ai_only', 'hybrid'];
        this.modeNames = {
            'human_only': 'Human Only',
            'ai_only': 'AI Only',
            'hybrid': 'Hybrid'
        };

        this.init();
    }

    // In the init() method, replace the connection test section:
async init() {
    // Initialize the application
    await this.loadTestEmails();
    await this.initializeModel();

    // Handle Google Sheets connection
    this.sheetsAvailable = false;

    if (window.googleSheetsAPI) {
        try {
            console.log('🔍 Testing Google Sheets connection...');

            // Test connection but don't wait too long
            const connectionPromise = window.googleSheetsAPI.testConnection();
            const timeoutPromise = new Promise(resolve => setTimeout(resolve, 3000));

            await Promise.race([connectionPromise, timeoutPromise]);

            this.sheetsAvailable = window.googleSheetsAPI.isAvailable;

            if (this.sheetsAvailable) {
                console.log('✅ Google Sheets connection successful');
            } else {
                console.warn('❌ Google Sheets connection failed - using backup storage');
            }
        } catch (error) {
            console.warn('❌ Google Sheets connection test failed:', error);
            this.sheetsAvailable = false;
        }
    } else {
        console.warn('❌ Google Sheets API not available - using backup storage');
        this.sheetsAvailable = false;
    }

    this.setupEventListeners();
    this.showScreen('landing');
}

    async loadTestEmails() {
        try {
            // Try multiple possible paths for the test emails
            const possiblePaths = [
                './data/processed/test_emails_selected.json',
                '../data/processed/test_emails_selected.json',
                './test_emails_selected.json'
            ];

            let emailsLoaded = false;

            for (const path of possiblePaths) {
                try {
                    const response = await fetch(path);
                    if (response.ok) {
                        const data = await response.json();
                        this.testEmails = data.emails;
                        console.log(`✅ Loaded ${this.testEmails.length} test emails from ${path}`);
                        emailsLoaded = true;
                        break;
                    }
                } catch (error) {
                    console.log(`❌ Failed to load from ${path}:`, error.message);
                    continue;
                }
            }

            if (!emailsLoaded) {
                throw new Error('Could not load test emails from any path');
            }

            // Ensure we have enough emails
            if (this.testEmails.length < 15) {
                console.warn(`Only ${this.testEmails.length} emails loaded, using mock data to supplement`);
                const mockEmails = this.getMockEmails();
                // Add mock emails if we don't have enough
                while (this.testEmails.length < 15) {
                    this.testEmails.push(...mockEmails);
                }
                // Trim to 15 if we have too many
                this.testEmails = this.testEmails.slice(0, 15);
            }

        } catch (error) {
            console.error('❌ Error loading test emails:', error);
            // Fallback to mock data for development
            console.log('🔄 Using mock emails for development');
            this.testEmails = this.getMockEmails();
            // Ensure we have exactly 15 emails
            while (this.testEmails.length < 15) {
                this.testEmails.push(...this.getMockEmails());
            }
            this.testEmails = this.testEmails.slice(0, 15);
        }
    }

    async initializeModel() {
        try {
            // Initialize the AI model
            await window.phishGuardModel.loadModel();
            this.updateLoadingProgress(100);

            // Wait a moment to show completion
            setTimeout(() => {
                document.getElementById('loadingStatus').textContent = 'Ready!';
            }, 500);

        } catch (error) {
            console.error('Error initializing model:', error);
            this.updateLoadingProgress(100, 'Model load failed - using fallback');
        }
    }

    updateLoadingProgress(percent, message = '') {
        const progressFill = document.querySelector('.progress-fill');
        const statusElement = document.getElementById('loadingStatus');

        if (progressFill) {
            progressFill.style.width = `${percent}%`;
        }

        if (statusElement) {
            statusElement.textContent = message || `Loading model: ${percent}%`;
        }
    }

    setupEventListeners() {
        // Landing screen
        document.getElementById('consentCheckbox').addEventListener('change', this.toggleStartButton.bind(this));
        document.getElementById('startStudyBtn').addEventListener('click', this.startStudy.bind(this));

        // Instructions screen
        document.getElementById('beginTestingBtn').addEventListener('click', this.beginTesting.bind(this));

        // Test screen
        document.getElementById('phishingBtn').addEventListener('click', () => this.makeDecision('phishing'));
        document.getElementById('safeBtn').addEventListener('click', () => this.makeDecision('safe'));
        document.getElementById('confidenceSlider').addEventListener('input', this.updateConfidenceDisplay.bind(this));
        document.getElementById('confirmBtn').addEventListener('click', this.confirmDecision.bind(this));

        // Survey screen
        document.getElementById('surveyForm').addEventListener('submit', this.submitSurvey.bind(this));
        this.setupRatingButtons();

        // Results screen
        document.getElementById('downloadDataBtn').addEventListener('click', this.downloadData.bind(this));
        document.getElementById('newSessionBtn').addEventListener('click', this.startNewSession.bind(this));
    }

    toggleStartButton() {
        const checkbox = document.getElementById('consentCheckbox');
        const button = document.getElementById('startStudyBtn');
        button.disabled = !checkbox.checked;
    }

    startStudy() {
        this.showScreen('instructions');
    }

    beginTesting() {
        this.setupTestSession();
        this.showScreen('test');
        this.showNextEmail();
    }

    setupTestSession() {
        // Shuffle emails and assign to modes
        this.shuffleArray(this.testEmails);

        console.log(`🔄 Setting up test session with ${this.testEmails.length} emails`);

        // Assign 5 emails to each mode (total 15)
        this.modeEmails = {
            'human_only': this.testEmails.slice(0, 5),
            'ai_only': this.testEmails.slice(5, 10),
            'hybrid': this.testEmails.slice(10, 15)
        };

        // Log distribution
        console.log('📊 Email distribution:');
        console.log('- Human Only:', this.modeEmails.human_only.length, 'emails');
        console.log('- AI Only:', this.modeEmails.ai_only.length, 'emails');
        console.log('- Hybrid:', this.modeEmails.hybrid.length, 'emails');

        // Reset counters
        this.currentModeIndex = 0;
        this.currentEmailIndex = 0;
        this.responses = [];

        // Shuffle the order of modes for each participant
        this.shuffleArray(this.modes);
        console.log('🔀 Mode order:', this.modes);
    }

    showNextEmail() {
        if (this.currentModeIndex >= this.modes.length) {
            this.showScreen('survey');
            return;
        }

        const currentMode = this.modes[this.currentModeIndex];
        const modeEmails = this.modeEmails[currentMode];

        if (this.currentEmailIndex >= modeEmails.length) {
            // Move to next mode
            this.currentModeIndex++;
            this.currentEmailIndex = 0;
            this.showNextEmail();
            return;
        }

        const email = modeEmails[this.currentEmailIndex];
        this.displayEmail(email, currentMode);
        this.startTimer();
        this.updateProgress();
    }

    displayEmail(email, mode) {
        // Update email content
        document.getElementById('emailContent').textContent = email.text;
        document.getElementById('emailSubject').textContent = email.subject || 'No Subject';

        // Update progress
        const totalProgress = (this.currentModeIndex * 5 + this.currentEmailIndex + 1) / 15 * 100;
        document.getElementById('progressFill').style.width = `${totalProgress}%`;
        document.getElementById('progressText').textContent =
            `Email ${this.currentModeIndex * 5 + this.currentEmailIndex + 1} of 15`;

        // Update mode indicator
        document.getElementById('modeIndicator').textContent = this.modeNames[mode];

        // Reset UI state
        this.resetDecisionUI();

        // Handle mode-specific UI
        this.setupModeUI(mode);

        // Store current email info
        this.currentEmail = email;
        this.currentMode = mode;
    }

    setupModeUI(mode) {
        const aiBox = document.getElementById('aiPredictionBox');
        const questions = document.getElementById('modeQuestions');
        const confirmBtn = document.getElementById('confirmBtn');

        // Hide all elements first
        aiBox.classList.add('hidden');
        questions.classList.add('hidden');
        confirmBtn.classList.add('hidden');

        document.getElementById('aiOnlyQuestion').classList.add('hidden');
        document.getElementById('hybridQuestion').classList.add('hidden');

        switch(mode) {
            case 'human_only':
                // No AI assistance, no additional questions
                break;

            case 'ai_only':
                // Show AI prediction after decision
                break;

            case 'hybrid':
                // Show AI prediction immediately
                this.showAIPrediction();
                aiBox.classList.remove('hidden');
                break;
        }
    }

    async showAIPrediction() {
        try {
            if (!this.currentEmail || !this.currentEmail.text) {
                console.error('No current email text available for AI prediction');
                return;
            }

            const prediction = await window.phishGuardModel.predict(this.currentEmail.text);

            const aiBox = document.getElementById('aiPredictionBox');
            const verdictElement = document.getElementById('aiVerdict');
            const confidenceElement = document.getElementById('aiConfidenceBadge');
            const confidenceFill = document.getElementById('confidenceFill');

            if (verdictElement && confidenceElement && confidenceFill) {
                verdictElement.textContent = prediction.prediction.toUpperCase();
                verdictElement.className = `verdict-text ${prediction.prediction === 'phishing' ? 'phishing-text' : 'safe-text'}`;

                const confidencePercent = Math.round(prediction.confidence * 100);
                confidenceElement.textContent = `${confidencePercent}% confident`;
                confidenceFill.style.width = `${confidencePercent}%`;

                // Store for later use
                this.currentAIPrediction = prediction;
            }

        } catch (error) {
            console.error('Error getting AI prediction:', error);
            // Use mock prediction as fallback
            this.currentAIPrediction = this.getMockPrediction(this.currentEmail.text);
            this.updateAIPredictionUI(this.currentAIPrediction);
        }
    }

    getMockPrediction(emailText) {
        // Simple heuristic-based mock prediction
        const text = emailText.toLowerCase();
        let phishingScore = 0;

        // Check for phishing indicators
        if (text.includes('password') || text.includes('login') || text.includes('account')) {
            phishingScore += 0.3;
        }
        if (text.includes('http') || text.includes('www.')) {
            phishingScore += 0.2;
        }
        if (text.includes('urgent') || text.includes('immediately')) {
            phishingScore += 0.2;
        }
        if (text.includes('money') || text.includes('payment')) {
            phishingScore += 0.3;
        }

        // Add some randomness for demo
        phishingScore += Math.random() * 0.1;
        phishingScore = Math.min(phishingScore, 0.95);

        const safeScore = 1 - phishingScore;
        const prediction = phishingScore > 0.5 ? 'phishing' : 'safe';
        const confidence = prediction === 'phishing' ? phishingScore : safeScore;

        return {
            prediction: prediction,
            confidence: confidence,
            probabilities: [safeScore, phishingScore]
        };
    }

    updateAIPredictionUI(prediction) {
        const aiBox = document.getElementById('aiPredictionBox');
        const verdictElement = document.getElementById('aiVerdict');
        const confidenceElement = document.getElementById('aiConfidenceBadge');
        const confidenceFill = document.getElementById('confidenceFill');

        if (verdictElement && confidenceElement && confidenceFill) {
            verdictElement.textContent = prediction.prediction.toUpperCase();
            verdictElement.className = `verdict-text ${prediction.prediction === 'phishing' ? 'phishing-text' : 'safe-text'}`;

            const confidencePercent = Math.round(prediction.confidence * 100);
            confidenceElement.textContent = `${confidencePercent}% confident`;
            confidenceFill.style.width = `${confidencePercent}%`;
        }
    }

    makeDecision(decision) {
        // Record the decision
        this.currentDecision = decision;

        // Highlight selected button
        document.getElementById('phishingBtn').classList.remove('active');
        document.getElementById('safeBtn').classList.remove('active');

        if (decision === 'phishing') {
            document.getElementById('phishingBtn').classList.add('active');
        } else {
            document.getElementById('safeBtn').classList.add('active');
        }

        // Show confidence slider and mode-specific questions
        this.showConfidenceSection();
    }

    showConfidenceSection() {
        const questions = document.getElementById('modeQuestions');
        const confirmBtn = document.getElementById('confirmBtn');

        confirmBtn.classList.remove('hidden');

        if (this.currentMode === 'ai_only') {
            // Show AI prediction and change-of-mind question
            this.showAIPrediction();
            document.getElementById('aiPredictionBox').classList.remove('hidden');
            questions.classList.remove('hidden');
            document.getElementById('aiOnlyQuestion').classList.remove('hidden');
        } else if (this.currentMode === 'hybrid') {
            // Show influence question
            questions.classList.remove('hidden');
            document.getElementById('hybridQuestion').classList.remove('hidden');
        }
    }

    confirmDecision() {
        this.stopTimer();

        const confidence = parseInt(document.getElementById('confidenceSlider').value);
        const response = {
            emailId: this.currentEmail.id,
            groundTruth: this.currentEmail.ground_truth,
            testMode: this.currentMode,
            humanDecision: this.currentDecision,
            humanConfidence: confidence,
            decisionTime: this.currentTimer,
            timestamp: new Date().toISOString()
        };

        // Add AI prediction if available
        if (this.currentAIPrediction) {
            response.aiPrediction = this.currentAIPrediction.prediction;
            response.aiConfidence = this.currentAIPrediction.confidence;
        }

        // Add mode-specific responses
        if (this.currentMode === 'ai_only') {
            const changeMind = document.querySelector('#aiOnlyQuestion .btn-secondary.active');
            response.changedMind = changeMind ? changeMind.dataset.answer === 'yes' : false;
        } else if (this.currentMode === 'hybrid') {
            const aiInfluence = document.querySelector('#hybridQuestion .btn-secondary.active');
            response.agreedWithAI = aiInfluence ? aiInfluence.dataset.answer === 'yes' : false;
        }

        this.responses.push(response);
        this.saveResponse(response);

        // Move to next email
        this.currentEmailIndex++;
        setTimeout(() => this.showNextEmail(), 500);
    }

    // Google Sheets integration functions
    async saveResponse(response) {
    try {
        console.log('Saving email response:', response);

        const saveData = {
            timestamp: response.timestamp,
            participant_id: this.participantId,
            session_id: this.sessionId,
            email_id: response.emailId,
            email_ground_truth: response.groundTruth,
            test_mode: response.testMode,
            human_decision: response.humanDecision,
            human_confidence: response.humanConfidence,
            ai_prediction: response.aiPrediction || '',
            ai_confidence: response.aiConfidence || 0,
            decision_time_seconds: response.decisionTime,
            agreed_with_ai: response.agreedWithAI || false,
            changed_mind: response.changedMind || false,
            user_feedback: response.user_feedback || '',
            country: this.getUserCountry(),
            device_type: this.getDeviceType()
        };

        // Try to save to Google Sheets
        let sheetsSuccess = false;
        if (window.googleSheetsAPI) {
            try {
                const sheetsResult = await window.googleSheetsAPI.saveToSheets(saveData);
                sheetsSuccess = sheetsResult.success;

                if (sheetsSuccess) {
                    console.log('✅ Data saved to Google Sheets');
                    this.showTemporaryMessage('Data saved successfully!', 'success');
                } else {
                    console.warn('❌ Google Sheets save failed, using backup');
                }
            } catch (sheetsError) {
                console.warn('Google Sheets save error:', sheetsError);
                sheetsSuccess = false;
            }
        }

        // Always save to localStorage as backup
        const localSuccess = this.saveToLocalStorage(saveData);

        if (!sheetsSuccess) {
            console.log('📦 Using localStorage backup');
            if (localSuccess) {
                this.showTemporaryMessage('Data saved locally (backup mode)', 'info');
            }
        }

    } catch (error) {
        console.error('Error saving response:', error);
        // Emergency fallback
        this.emergencySave(response);
    }
}

    async saveToGoogleSheets(data) {
        try {
            if (!window.googleSheetsAPI) {
                console.warn('Google Sheets API not available');
                return false;
            }

            const result = await window.googleSheetsAPI.saveToSheets(data);
            return result.success;

        } catch (error) {
            console.error('Error saving to Google Sheets:', error);
            return false;
        }
    }

    saveToLocalStorage(data) {
        try {
            const key = `phishguard_${this.participantId}_${Date.now()}`;
            const existing = JSON.parse(localStorage.getItem('phishguard_backup') || '[]');
            existing.push({ key, data, timestamp: new Date().toISOString() });
            localStorage.setItem('phishguard_backup', JSON.stringify(existing));
            console.log('✅ Data saved to localStorage backup');

            // Show message to user
            return true;
        } catch (error) {
            console.error('Error saving to localStorage:', error);
            return false;
        }
    }

    showTemporaryMessage(message, type = 'info') {
        // Remove any existing messages
        const existingMessages = document.querySelectorAll('.temp-message');
        existingMessages.forEach(msg => msg.remove());

        const messageElement = document.createElement('div');
        messageElement.textContent = message;
        messageElement.className = 'temp-message';
        messageElement.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: ${type === 'success' ? 'var(--accent-green)' : type === 'error' ? 'var(--accent-red)' : 'var(--accent-cyan)'};
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            font-weight: bold;
            box-shadow: var(--shadow-card);
            max-width: 300px;
        `;
        document.body.appendChild(messageElement);

        setTimeout(() => {
            if (document.body.contains(messageElement)) {
                document.body.removeChild(messageElement);
            }
        }, 4000);
    }

    startTimer() {
        this.currentTimer = 0;
        this.emailStartTime = Date.now();
        this.updateTimerDisplay();

        this.timerInterval = setInterval(() => {
            this.currentTimer = (Date.now() - this.emailStartTime) / 1000;
            this.updateTimerDisplay();
        }, 100);
    }

    stopTimer() {
        if (this.timerInterval) {
            clearInterval(this.timerInterval);
            this.timerInterval = null;
        }
    }

    updateTimerDisplay() {
        const timerElement = document.getElementById('timerDisplay');
        if (timerElement) {
            timerElement.textContent = `${this.currentTimer.toFixed(1)}s`;
        }
    }

    updateProgress() {
        const totalEmails = 15;
        const completed = this.currentModeIndex * 5 + this.currentEmailIndex;
        const percent = (completed / totalEmails) * 100;

        document.getElementById('progressFill').style.width = `${percent}%`;
        document.getElementById('progressText').textContent = `Email ${completed + 1} of ${totalEmails}`;
    }

    updateConfidenceDisplay() {
        const slider = document.getElementById('confidenceSlider');
        const value = document.getElementById('confidenceValue');
        value.textContent = slider.value;
    }

    resetDecisionUI() {
        // Reset buttons
        document.getElementById('phishingBtn').classList.remove('active');
        document.getElementById('safeBtn').classList.remove('active');

        // Reset slider
        document.getElementById('confidenceSlider').value = 3;
        this.updateConfidenceDisplay();

        // Hide additional UI
        document.getElementById('confirmBtn').classList.add('hidden');
        document.getElementById('modeQuestions').classList.add('hidden');
        document.getElementById('aiPredictionBox').classList.add('hidden');

        // Clear any active question buttons
        document.querySelectorAll('.question-buttons .btn-secondary.active').forEach(btn => {
            btn.classList.remove('active');
        });
    }

    setupRatingButtons() {
        document.querySelectorAll('.rating-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                document.querySelectorAll('.rating-btn').forEach(b => b.classList.remove('active'));
                e.target.classList.add('active');
                document.getElementById('aiHelpfulness').value = e.target.dataset.value;
            });
        });

        // Setup question buttons
        document.querySelectorAll('.question-buttons .btn-secondary').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.target.parentElement.querySelectorAll('.btn-secondary').forEach(b => {
                    b.classList.remove('active');
                });
                e.target.classList.add('active');
            });
        });
    }

    async submitSurvey(e) {
        e.preventDefault();

        const formData = new FormData(e.target);
        const surveyData = {
            ageRange: document.getElementById('ageRange').value,
            techExpertise: document.getElementById('techExpertise').value,
            aiHelpfulness: document.getElementById('aiHelpfulness').value,
            trustAI: document.querySelector('input[name="trustAI"]:checked')?.value,
            userFeedback: document.getElementById('userFeedback').value
        };

        // Save survey data with the last response
        if (this.responses.length > 0) {
            const lastResponse = this.responses[this.responses.length - 1];
            await this.saveResponse({
                ...lastResponse,
                user_feedback: surveyData.userFeedback
            });
        }

        this.showResults();
    }

    showResults() {
        this.calculateResults();
        this.showScreen('results');
    }

    calculateResults() {
        const total = this.responses.length;
        const correct = this.responses.filter(r => r.humanDecision === r.groundTruth).length;
        const userAccuracy = total > 0 ? (correct / total) * 100 : 0;

        // For demo, using mock AI accuracy - replace with actual calculations
        const aiAccuracy = 92;
        const agreementRate = 78;

        document.getElementById('userAccuracy').textContent = `${Math.round(userAccuracy)}%`;
        document.getElementById('aiAccuracy').textContent = `${aiAccuracy}%`;
        document.getElementById('agreementRate').textContent = `${agreementRate}%`;

        // Update performance breakdown
        this.updatePerformanceBreakdown();
    }

    updatePerformanceBreakdown() {
        const breakdown = document.getElementById('performanceBreakdown');

        // Calculate mode-specific performance
        const modePerformance = {};
        this.modes.forEach(mode => {
            const modeResponses = this.responses.filter(r => r.testMode === mode);
            const correct = modeResponses.filter(r => r.humanDecision === r.groundTruth).length;
            modePerformance[mode] = modeResponses.length > 0 ? (correct / modeResponses.length) * 100 : 0;
        });

        breakdown.innerHTML = `
            <h4>Performance by Test Mode:</h4>
            <div class="mode-performance">
                <div>Human Only: <strong>${Math.round(modePerformance.human_only)}%</strong> accuracy</div>
                <div>AI Only: <strong>${Math.round(modePerformance.ai_only)}%</strong> accuracy</div>
                <div>Hybrid: <strong>${Math.round(modePerformance.hybrid)}%</strong> accuracy</div>
            </div>
            <p style="margin-top: 15px; color: var(--text-secondary);">
                Your data has been recorded anonymously. Thank you for contributing to cybersecurity research!
            </p>
        `;
    }

    downloadData() {
        const data = {
            participantId: this.participantId,
            sessionId: this.sessionId,
            responses: this.responses,
            downloadTime: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `phishguard_data_${this.participantId}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    startNewSession() {
        this.sessionId = this.generateUUID();
        this.showScreen('landing');
    }

    showScreen(screenName) {
        // Hide all screens
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });

        // Show target screen
        document.getElementById(`${screenName}Screen`).classList.add('active');
        this.currentScreen = screenName;
    }

    // Utility methods
    generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
            const r = Math.random() * 16 | 0;
            const v = c == 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    shuffleArray(array) {
        for (let i = array.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [array[i], array[j]] = [array[j], array[i]];
        }
        return array;
    }

    getUserCountry() {
        // Simple country detection - in production, use a proper geolocation service
        const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
        if (timezone.includes('America')) return 'US';
        if (timezone.includes('Europe')) return 'EU';
        if (timezone.includes('Asia')) return 'AS';
        return 'Unknown';
    }

    getDeviceType() {
        const ua = navigator.userAgent;
        if (/(tablet|ipad|playbook|silk)|(android(?!.*mobi))/i.test(ua)) {
            return 'Tablet';
        }
        if (/Mobile|iP(hone|od)|Android|BlackBerry|IEMobile|Kindle|Silk-Accelerated|(hpw|web)OS|Opera M(obi|ini)/.test(ua)) {
            return 'Mobile';
        }
        return 'Desktop';
    }

    getMockEmails() {
        // Fallback mock emails for development
        return [
            {
                id: 'mock_001',
                text: 'Dear user, your account has been compromised. Click here to verify your identity: http://fake-security.com/verify',
                subject: 'Urgent: Account Security Alert',
                ground_truth: 'phishing',
                difficulty: 'medium'
            },
            {
                id: 'mock_002',
                text: 'Hi team, attached is the Q3 report for review. Please provide feedback by Friday. Best regards, Management',
                subject: 'Q3 Performance Report',
                ground_truth: 'safe',
                difficulty: 'easy'
            }
            // Add more mock emails as needed
        ];
    }
}

// Initialize the application when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.phishGuardApp = new PhishGuardApp();
});

// Make sure these functions are available globally
function getParticipantId() {
    return window.phishGuardApp?.participantId || 'unknown';
}

function getSessionId() {
    return window.phishGuardApp?.sessionId || 'unknown';
}

function getUserCountry() {
    return window.phishGuardApp?.getUserCountry() || 'unknown';
}

function getDeviceType() {
    return window.phishGuardApp?.getDeviceType() || 'unknown';
}