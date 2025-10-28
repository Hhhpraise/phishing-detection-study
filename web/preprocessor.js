// PhishGuard - Text Preprocessor
class PhishGuardPreprocessor {
    constructor(vocabulary, idfValues, engineeredFeatures) {
        this.vocabulary = vocabulary || {};
        this.idfValues = idfValues || [];
        this.engineeredFeatures = engineeredFeatures || [];
        this.vocabularySize = Object.keys(this.vocabulary).length;
        this.engineeredFeaturesSize = this.engineeredFeatures.length;

        console.log(`✅ Preprocessor initialized: ${this.vocabularySize} words, ${this.engineeredFeaturesSize} engineered features`);
    }

    // Clean and tokenize email text
    cleanText(text) {
        if (!text) return '';

        return text.toLowerCase()
            .replace(/<[^>]*>/g, ' ') // Remove HTML tags
            .replace(/https?:\/\/[^\s]+/g, 'URL_PLACEHOLDER') // Replace URLs
            .replace(/www\.[^\s]+/g, 'URL_PLACEHOLDER')
            .replace(/[^\w\s]/g, ' ') // Remove punctuation
            .replace(/\s+/g, ' ') // Normalize whitespace
            .trim();
    }

    // Tokenize text into words
    tokenize(text) {
        return text.split(/\s+/).filter(word => word.length > 1);
    }

    // Extract engineered features from email text
    extractFeatures(text) {
        const features = new Array(this.engineeredFeaturesSize).fill(0);

        if (!text) return features;

        const cleanText = this.cleanText(text);
        const tokens = this.tokenize(cleanText);

        // Email length features
        features[0] = text.length; // email_length
        features[1] = tokens.length; // word_count

        // URL count
        const urlCount = (text.match(/https?:\/\/|www\./g) || []).length;
        features[2] = urlCount; // url_count
        features[3] = urlCount > 0 ? 1 : 0; // has_urls

        // Urgent language features
        const urgentWords = ['urgent', 'immediately', 'asap', 'emergency', 'important', 'alert'];
        let urgentCount = 0;
        urgentWords.forEach(word => {
            if (cleanText.includes(word)) urgentCount++;
        });
        features[4] = urgentCount; // urgent_word_count
        features[5] = urgentCount > 0 ? 1 : 0; // has_urgent_words

        // Money-related features
        const moneyWords = ['money', 'payment', 'bank', 'credit', 'card', 'price', 'cost', '$'];
        let moneyCount = 0;
        moneyWords.forEach(word => {
            if (cleanText.includes(word)) moneyCount++;
        });
        features[6] = moneyCount; // money_word_count
        features[7] = moneyCount > 0 ? 1 : 0; // has_money_words

        // Punctuation features
        features[8] = (text.match(/!/g) || []).length; // exclamation_count
        features[9] = (text.match(/\?/g) || []).length; // question_count

        return features;
    }

    // Create TF-IDF vector from tokens
    createTfIdfVector(tokens) {
        const vector = new Array(this.vocabularySize).fill(0);

        if (!tokens || tokens.length === 0) return vector;

        // Calculate term frequencies
        const termFreq = {};
        tokens.forEach(token => {
            termFreq[token] = (termFreq[token] || 0) + 1;
        });

        // Apply TF-IDF
        Object.keys(termFreq).forEach(token => {
            const index = this.vocabulary[token];
            if (index !== undefined && index < this.vocabularySize) {
                const tf = termFreq[token] / tokens.length;
                const idf = this.idfValues[index] || 1.0; // Fallback to 1.0 if IDF not available
                vector[index] = tf * idf;
            }
        });

        return vector;
    }

    // Main preprocessing function
    preprocessEmail(emailText) {
        try {
            if (!emailText || typeof emailText !== 'string') {
                console.warn('Invalid email text provided to preprocessor');
                return this.getFallbackFeatures();
            }

            // Clean and tokenize
            const cleanText = this.cleanText(emailText);
            const tokens = this.tokenize(cleanText);

            // Create TF-IDF vector
            const tfidfVector = this.createTfIdfVector(tokens);

            // Extract engineered features
            const engineeredFeatures = this.extractFeatures(emailText);

            // Combine features
            const combinedFeatures = [...tfidfVector, ...engineeredFeatures];

            // Normalize features to prevent extreme values
            return this.normalizeFeatures(combinedFeatures);

        } catch (error) {
            console.error('Error in preprocessEmail:', error);
            return this.getFallbackFeatures();
        }
    }

    // Normalize features to 0-1 range
    normalizeFeatures(features) {
        const normalized = [...features];
        const maxVal = Math.max(...normalized.map(Math.abs));

        if (maxVal > 0) {
            for (let i = 0; i < normalized.length; i++) {
                normalized[i] = normalized[i] / maxVal;
            }
        }

        return normalized;
    }

    // Fallback features when preprocessing fails
    getFallbackFeatures() {
        const totalFeatures = this.vocabularySize + this.engineeredFeaturesSize;
        return new Array(totalFeatures).fill(0.1); // Small non-zero values
    }
}

// Create global instance if needed
if (typeof window !== 'undefined') {
    window.PhishGuardPreprocessor = PhishGuardPreprocessor;
}