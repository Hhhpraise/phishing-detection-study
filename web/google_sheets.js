// Google Sheets API Configuration - SIMPLIFIED & RELIABLE
const GOOGLE_SCRIPT_URL = 'https://script.google.com/macros/s/AKfycbzu0WAmqhaKU_iv2cyvQIFmftJy4A4_eH290j2jPf4YqBsnOQs-dT7EiNuqVDfb9w2c7w/exec';

class GoogleSheetsAPI {
    constructor() {
        this.scriptURL = GOOGLE_SCRIPT_URL;
        this.isAvailable = true; // Assume it's available, we'll test
        this.connectionTested = false;
    }

    // Simple connection test that always uses no-cors
    async testConnection() {
        try {
            console.log('🔍 Testing Google Sheets connection...');

            // Use no-cors for testing (avoids CORS issues)
            await fetch(this.scriptURL, {
                method: 'GET',
                mode: 'no-cors'
            });

            this.isAvailable = true;
            this.connectionTested = true;
            console.log('✅ Google Sheets connection successful');
            return true;

        } catch (error) {
            console.error('❌ Google Sheets connection failed:', error);
            this.isAvailable = false;
            this.connectionTested = true;
            return false;
        }
    }

    // Save data to Google Sheets - always use no-cors
    async saveToSheets(data) {
        // If we haven't tested connection yet, assume it's available
        if (!this.connectionTested) {
            await this.testConnection();
        }

        if (!this.isAvailable) {
            console.warn('Google Sheets not available, skipping save');
            return { success: false, error: 'Google Sheets not available' };
        }

        try {
            console.log('🔄 Saving to Google Sheets:', data);

            // Format data as array matching your sheet columns
            const rowData = [
                data.timestamp || new Date().toISOString(),
                data.participant_id || '',
                data.session_id || '',
                data.email_id || '',
                data.email_ground_truth || '',
                data.test_mode || '',
                data.human_decision || '',
                data.human_confidence || '',
                data.ai_prediction || '',
                data.ai_confidence || 0,
                data.decision_time_seconds || 0,
                data.agreed_with_ai || false,
                data.changed_mind || false,
                data.user_feedback || '',
                data.country || '',
                data.device_type || ''
            ];

            console.log('Formatted row data:', rowData);

            // Always use no-cors to avoid CORS issues
            await fetch(this.scriptURL, {
                method: 'POST',
                mode: 'no-cors',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    action: 'addRow',
                    data: rowData
                })
            });

            console.log('✅ Data sent to Google Sheets (no-cors mode)');
            return { success: true };

        } catch (error) {
            console.error('❌ Error saving to Google Sheets:', error);
            this.isAvailable = false;
            return { success: false, error: error.toString() };
        }
    }

    isSheetsAvailable() {
        return this.isAvailable;
    }
}

// Create global instance
window.googleSheetsAPI = new GoogleSheetsAPI();