/**
 * Representation Power of MLPs - API Communication Client
 */
const API = {
    /**
     * Sends network configuration parameters to Flask backend to train a model.
     * @param {Object} params Config parameters
     * @param {string} params.dataset_name Name of dataset ('xor', 'spiral', 'moons', 'circles')
     * @param {Array<number>} params.layers Hidden layer dimensions (e.g. [4, 4])
     * @param {string} params.activation Activation name ('relu', 'sigmoid', 'tanh')
     * @param {number} params.learning_rate Learning rate
     * @param {number} params.epochs Number of epochs
     * @param {number} params.noise Dataset noise scale
     * @returns {Promise<Object>} Backend response with dataset, grid details and epoch records
     */
    async trainNetwork(params) {
        try {
            const response = await fetch('/api/train', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(params)
            });
            
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server error: ${response.status}`);
            }
            
            return await response.json();
        } catch (error) {
            console.error("API training request failed:", error);
            throw error;
        }
    }
};

// Export to global scope
window.API = API;
