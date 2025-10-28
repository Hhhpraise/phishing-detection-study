// PhishGuard - ONNX.js Model Interface
// This file handles model loading and predictions using ONNX.js

class PhishGuardModel {
    constructor() {
        this.session = null;
        this.preprocessor = null;
        this.isLoaded = false;
        this.useONNX = false; // Set to true to use ONNX model (requires conversion)
    }

    // Load model and preprocessing data
    async loadModel() {
        try {
            // Load vocabulary and preprocessing data
            let vocabData;
            try {
                const response = await fetch('./model/vocabulary.json');
                if (response.ok) {
                    vocabData = await response.json();
                    console.log('✅ Vocabulary loaded:', Object.keys(vocabData.vocabulary).length, 'words');
                } else {
                    throw new Error('Vocabulary file not found');
                }
            } catch (error) {
                console.warn('⚠️ Could not load vocabulary.json, using mock vocabulary:', error);
                vocabData = this.createMockVocabulary();
            }

            // Initialize preprocessor
            this.preprocessor = new PhishGuardPreprocessor(
                vocabData.vocabulary,
                vocabData.idfValues,
                vocabData.engineeredFeatures
            );

            // Try to load ONNX model if available
            if (this.useONNX && typeof ort !== 'undefined') {
                try {
                    console.log('🔄 Loading ONNX model...');
                    this.session = await ort.InferenceSession.create('./model/phishing_model.onnx');
                    console.log('✅ ONNX model loaded successfully');
                } catch (error) {
                    console.warn('⚠️ ONNX model not found, using enhanced heuristics:', error);
                    this.session = null;
                }
            }

            this.isLoaded = true;
            return true;

        } catch (error) {
            console.error('❌ Error loading model:', error);
            // Create fallback preprocessor with mock data
            const mockVocab = this.createMockVocabulary();
            this.preprocessor = new PhishGuardPreprocessor(
                mockVocab.vocabulary,
                mockVocab.idfValues,
                mockVocab.engineeredFeatures
            );
            this.isLoaded = true;
            return true;
        }
    }

    createMockVocabulary() {
        // Enhanced vocabulary for better detection
        const phishingWords = [
            'urgent', 'password', 'account', 'verify', 'login', 'security',
            'suspended', 'locked', 'expire', 'update', 'confirm', 'click',
            'link', 'immediately', 'action', 'required', 'alert', 'warning',
            'prize', 'winner', 'free', 'gift', 'money', 'payment', 'bank',
            'credit', 'card', 'social', 'security', 'ssn', 'tax', 'refund',
            'limited', 'time', 'offer', 'congratulations', 'selected', 'claim',
            'restore', 'access', 'blocked', 'unusual', 'activity', 'suspicious'
        ];

        const safeWords = [
            'meeting', 'schedule', 'report', 'document', 'team', 'project',
            'discussion', 'review', 'feedback', 'update', 'summary', 'proposal',
            'agenda', 'minutes', 'presentation', 'data', 'analysis', 'results',
            'thanks', 'regards', 'sincerely', 'best', 'hello', 'hi', 'dear'
        ];

        const vocabulary = {};
        const idfValues = [];
        const allWords = [...phishingWords, ...safeWords];

        allWords.forEach((word, index) => {
            vocabulary[word] = index;
            // Higher IDF for phishing words
            idfValues[index] = phishingWords.includes(word) ? 2.0 : 1.0;
        });

        const engineeredFeatures = [
            'email_length', 'word_count', 'url_count', 'has_urls',
            'urgent_word_count', 'has_urgent_words', 'money_word_count',
            'has_money_words', 'exclamation_count', 'question_count'
        ];

        return {
            vocabulary: vocabulary,
            idfValues: idfValues,
            engineeredFeatures: engineeredFeatures,
            input_dim: allWords.length + engineeredFeatures.length
        };
    }

    // Make prediction on email text
    // In the predict method, add better error handling:
async predict(emailText) {
    if (!this.isLoaded) {
        throw new Error('Model not loaded. Call loadModel() first.');
    }

    try {
        // Validate input
        if (!emailText || typeof emailText !== 'string') {
            console.warn('Invalid email text provided for prediction');
            return this.getFallbackPrediction();
        }

        // Preprocess the email with error handling
        let features;
        try {
            features = this.preprocessor.preprocessEmail(emailText);
        } catch (preprocessError) {
            console.error('Preprocessing failed, using heuristic prediction:', preprocessError);
            return this.enhancedHeuristicPrediction(emailText);
        }

        // Validate features
        if (!features || !Array.isArray(features) || features.length === 0) {
            console.warn('Invalid features generated, using heuristic prediction');
            return this.enhancedHeuristicPrediction(emailText);
        }

        // Use ONNX model if available
        if (this.session) {
            try {
                return await this.predictWithONNX(features);
            } catch (onnxError) {
                console.error('ONNX prediction failed, using heuristic:', onnxError);
                return this.enhancedHeuristicPrediction(emailText);
            }
        } else {
            // Use enhanced heuristic prediction
            return this.enhancedHeuristicPrediction(emailText);
        }

    } catch (error) {
        console.error('Error making prediction:', error);
        return this.getFallbackPrediction();
    }
}

// Add fallback prediction method
getFallbackPrediction() {
    return {
        prediction: 'safe',
        confidence: 0.5,
        probabilities: [0.5, 0.5],
        source: 'fallback'
    };
}

    async predictWithONNX(features) {
        try {
            // Convert features to ONNX tensor
            const inputTensor = new ort.Tensor('float32', features, [1, features.length]);

            // Run inference
            const results = await this.session.run({ input: inputTensor });
            const output = results.output.data;

            // Get probabilities (assuming output is [safe_prob, phishing_prob])
            const safeProb = output[0];
            const phishingProb = output[1];

            const prediction = phishingProb > 0.5 ? 'phishing' : 'safe';
            const confidence = prediction === 'phishing' ? phishingProb : safeProb;

            return {
                prediction: prediction,
                confidence: confidence,
                probabilities: [safeProb, phishingProb],
                source: 'onnx'
            };
        } catch (error) {
            console.error('ONNX prediction error:', error);
            return this.enhancedHeuristicPrediction(emailText);
        }
    }

    // Enhanced heuristic-based prediction with better accuracy
    enhancedHeuristicPrediction(emailText) {
        const text = emailText.toLowerCase();
        let phishingScore = 0;

        // Strong phishing indicators (high weight)
        const strongIndicators = {
            'verify your account': 0.4,
            'suspended': 0.3,
            'click here': 0.3,
            'update your password': 0.4,
            'confirm your identity': 0.4,
            'unusual activity': 0.3,
            'your account will be': 0.3,
            'limited time': 0.2,
            'act now': 0.2,
            'urgent': 0.2,
            'immediately': 0.2
        };

        // Medium indicators
        const mediumIndicators = {
            'password': 0.15,
            'login': 0.15,
            'security': 0.1,
            'bank': 0.15,
            'credit card': 0.15,
            'prize': 0.15,
            'winner': 0.15,
            'free': 0.1
        };

        // Check strong indicators
        for (const [phrase, weight] of Object.entries(strongIndicators)) {
            if (text.includes(phrase)) {
                phishingScore += weight;
            }
        }

        // Check medium indicators
        for (const [phrase, weight] of Object.entries(mediumIndicators)) {
            if (text.includes(phrase)) {
                phishingScore += weight;
            }
        }

        // URL analysis
        const urlCount = (text.match(/https?:\/\/|www\./g) || []).length;
        if (urlCount > 2) {
            phishingScore += 0.2;
        } else if (urlCount > 0) {
            phishingScore += 0.1;
        }

        // Suspicious patterns
        if (/\d{3}[-.\s]?\d{2}[-.\s]?\d{4}/.test(text)) { // SSN pattern
            phishingScore += 0.3;
        }
        if (text.includes('$') || text.includes('money')) {
            phishingScore += 0.1;
        }
        if ((text.match(/!/g) || []).length > 2) {
            phishingScore += 0.1;
        }

        // Safe indicators (reduce score)
        const safeIndicators = ['meeting', 'schedule', 'report', 'team', 'project', 'discussion'];
        for (const word of safeIndicators) {
            if (text.includes(word)) {
                phishingScore -= 0.1;
            }
        }

        // Normalize score
        phishingScore = Math.max(0, Math.min(phishingScore, 1));

        // Add small randomness for variety
        phishingScore += (Math.random() - 0.5) * 0.05;
        phishingScore = Math.max(0, Math.min(phishingScore, 1));

        const safeScore = 1 - phishingScore;
        const prediction = phishingScore > 0.5 ? 'phishing' : 'safe';
        const confidence = prediction === 'phishing' ? phishingScore : safeScore;

        return {
            prediction: prediction,
            confidence: confidence,
            probabilities: [safeScore, phishingScore],
            source: 'heuristic'
        };
    }

    // Get model status
    getStatus() {
        return {
            loaded: this.isLoaded,
            usingONNX: this.session !== null,
            inputDimension: this.preprocessor ?
                this.preprocessor.vocabularySize + this.preprocessor.engineeredFeatures.length : 0
        };
    }
}

// Create global instance
window.phishGuardModel = new PhishGuardModel();

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PhishGuardModel;
}