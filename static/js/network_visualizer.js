/**
 * Representation Power of MLPs - Network Visualizer
 * HTML5 Canvas renderer for the MLP architecture with animated pulses
 */
class NetworkVisualizer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        
        this.topology = [2, 4, 1]; // Default topology
        this.weights = null;       // 3D list: weights[layer][input_neuron][output_neuron]
        this.biases = null;        // 2D list: biases[layer][neuron]
        this.activations = null;   // 2D list: activations[layer][neuron] (live values)
        this.activationName = 'relu';
        
        this.nodes = [];           // Cache node coordinates: nodes[layer][neuron] = {x, y, radius}
        this.hoveredNode = null;   // {layerIndex, nodeIndex}
        
        this.pulseOffset = 0;      // Animate feedforward pulses procedurally
        this.animationFrameId = null;
        
        this.init();
    }
    
    init() {
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Track mouse movement for hovering info
        this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        this.canvas.addEventListener('mouseleave', () => {
            this.hoveredNode = null;
            this.draw();
        });
        
        // Start animation loop
        this.animate();
    }
    
    resize() {
        if (!this.canvas) return;
        const rect = this.canvas.parentElement.getBoundingClientRect();
        this.canvas.width = rect.width * window.devicePixelRatio;
        this.canvas.height = Math.max(300, rect.height) * window.devicePixelRatio;
        this.canvas.style.width = '100%';
        this.canvas.style.height = '100%';
        this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
        this.calculateNodePositions();
        this.draw();
    }
    
    setTopology(layers, activationName = 'relu') {
        // layers: list of hidden neurons, e.g. [4, 4] -> complete topology is [2, 4, 4, 1]
        this.topology = [2, ...layers, 1];
        this.activationName = activationName;
        this.calculateNodePositions();
        this.hoveredNode = null;
        this.draw();
    }
    
    setWeightsAndBiases(weights, biases) {
        this.weights = weights;
        this.biases = biases;
        this.draw();
    }
    
    setLiveActivations(activations) {
        this.activations = activations;
        this.draw();
    }
    
    calculateNodePositions() {
        const width = this.canvas.width / window.devicePixelRatio;
        const height = this.canvas.height / window.devicePixelRatio;
        
        this.nodes = [];
        const layerCount = this.topology.length;
        const hSpacing = width / (layerCount + 0.6);
        const startX = hSpacing * 0.8;
        
        for (let l = 0; l < layerCount; l++) {
            const nodeCount = this.topology[l];
            const layerNodes = [];
            const vSpacing = height / (nodeCount + 1);
            const x = startX + l * hSpacing;
            
            for (let n = 0; n < nodeCount; n++) {
                const y = vSpacing * (n + 1);
                layerNodes.push({
                    x: x,
                    y: y,
                    radius: l === 0 || l === layerCount - 1 ? 16 : 14
                });
            }
            this.nodes.push(layerNodes);
        }
    }
    
    handleMouseMove(e) {
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        let found = null;
        for (let l = 0; l < this.nodes.length; l++) {
            for (let n = 0; n < this.nodes[l].length; n++) {
                const node = this.nodes[l][n];
                const dist = Math.hypot(mouseX - node.x, mouseY - node.y);
                if (dist <= node.radius) {
                    found = { layerIndex: l, nodeIndex: n };
                    break;
                }
            }
            if (found) break;
        }
        
        if (JSON.stringify(this.hoveredNode) !== JSON.stringify(found)) {
            this.hoveredNode = found;
            this.draw();
        }
    }
    
    // JS implementation of activation functions for live tooltips / forward passes
    activationDesc(val) {
        if (this.activationName === 'relu') {
            return `ReLU(z) = max(0, ${val.toFixed(2)}) = ${Math.max(0, val).toFixed(2)}`;
        } else if (this.activationName === 'sigmoid') {
            const sig = 1 / (1 + Math.exp(-val));
            return `Sigmoid(z) = ${sig.toFixed(2)}`;
        } else if (this.activationName === 'tanh') {
            return `Tanh(z) = ${Math.tanh(val).toFixed(2)}`;
        }
        return val.toFixed(2);
    }
    
    draw() {
        if (!this.canvas) return;
        const width = this.canvas.width / window.devicePixelRatio;
        const height = this.canvas.height / window.devicePixelRatio;
        
        this.ctx.clearRect(0, 0, width, height);
        
        // 1. Draw Connections (Weights)
        for (let l = 0; l < this.nodes.length - 1; l++) {
            const currentLayerNodes = this.nodes[l];
            const nextLayerNodes = this.nodes[l + 1];
            
            for (let i = 0; i < currentLayerNodes.length; i++) {
                for (let j = 0; j < nextLayerNodes.length; j++) {
                    const nodeA = currentLayerNodes[i];
                    const nodeB = nextLayerNodes[j];
                    
                    let weightVal = 0;
                    let hasWeights = false;
                    
                    if (this.weights && this.weights[l] && this.weights[l][i]) {
                        weightVal = this.weights[l][i][j];
                        hasWeights = true;
                    }
                    
                    // Style connection based on weight
                    this.ctx.beginPath();
                    this.ctx.moveTo(nodeA.x, nodeA.y);
                    this.ctx.lineTo(nodeB.x, nodeB.y);
                    
                    if (hasWeights) {
                        const magnitude = Math.min(Math.abs(weightVal), 5); // Clip visual representation at magnitude 5
                        this.ctx.lineWidth = 0.5 + magnitude * 1.5;
                        
                        if (weightVal > 0) {
                            // Positive weight: Neon blue
                            this.ctx.strokeStyle = `rgba(99, 102, 241, ${0.15 + (magnitude / 5) * 0.55})`;
                        } else {
                            // Negative weight: Neon orange/red
                            this.ctx.strokeStyle = `rgba(249, 115, 22, ${0.15 + (magnitude / 5) * 0.55})`;
                        }
                    } else {
                        // Default initial weights before training
                        this.ctx.lineWidth = 1;
                        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
                    }
                    this.ctx.stroke();
                    
                    // Draw flowing signal pulses along connection
                    if (hasWeights && Math.abs(weightVal) > 0.1) {
                        const pathLength = Math.hypot(nodeB.x - nodeA.x, nodeB.y - nodeA.y);
                        // Speed proportional to weight magnitude, or constant
                        const speedFactor = 0.005 * (0.8 + Math.abs(weightVal) * 0.2);
                        
                        // Procedural animated pulse
                        const t = (this.pulseOffset * speedFactor + (i * 0.23 + j * 0.17)) % 1.0;
                        const pulseX = nodeA.x + t * (nodeB.x - nodeA.x);
                        const pulseY = nodeA.y + t * (nodeB.y - nodeA.y);
                        
                        this.ctx.beginPath();
                        this.ctx.arc(pulseX, pulseY, 2.5, 0, 2 * Math.PI);
                        this.ctx.fillStyle = weightVal > 0 ? '#6366f1' : '#f97316';
                        this.ctx.fill();
                    }
                }
            }
        }
        
        // 2. Draw Nodes (Neurons)
        for (let l = 0; l < this.nodes.length; l++) {
            const layerNodes = this.nodes[l];
            for (let n = 0; n < layerNodes.length; n++) {
                const node = layerNodes[n];
                
                // Get node properties
                let activationVal = 0;
                let hasActivations = false;
                if (this.activations && this.activations[l] !== undefined && this.activations[l][n] !== undefined) {
                    activationVal = this.activations[l][n];
                    hasActivations = true;
                }
                
                let biasVal = 0;
                let hasBiases = false;
                if (l > 0 && this.biases && this.biases[l-1] && this.biases[l-1][n] !== undefined) {
                    biasVal = this.biases[l-1][n];
                    hasBiases = true;
                }
                
                // Determine node colors based on activations
                // High activation = glowing teal/cyan, low activation = dark blue
                let fillColor = 'rgba(10, 10, 26, 0.9)';
                let glowColor = 'rgba(255, 255, 255, 0.15)';
                let strokeColor = 'rgba(255, 255, 255, 0.2)';
                
                if (hasActivations) {
                    // Normalize activation visually (usually between 0 and 1 for sigmoid, or 0 and >1 for ReLU)
                    const normAct = Math.min(Math.max(activationVal, 0), 1);
                    fillColor = `rgba(20, 184, 166, ${0.1 + normAct * 0.45})`; // Teal glow
                    glowColor = `rgba(20, 184, 166, ${0.2 + normAct * 0.6})`;
                    strokeColor = `rgba(20, 184, 166, ${0.4 + normAct * 0.6})`;
                }
                
                // Neon glow around active neurons
                this.ctx.shadowBlur = hasActivations && activationVal > 0.1 ? 8 * Math.min(activationVal, 1.5) : 0;
                this.ctx.shadowColor = glowColor;
                
                // Node circle background
                this.ctx.beginPath();
                this.ctx.arc(node.x, node.y, node.radius, 0, 2 * Math.PI);
                this.ctx.fillStyle = fillColor;
                this.ctx.fill();
                
                // Node border
                this.ctx.shadowBlur = 0; // Turn off shadows for cleaner borders
                this.ctx.lineWidth = 2;
                this.ctx.strokeStyle = strokeColor;
                this.ctx.stroke();
                
                // Node label text
                this.ctx.font = '600 10px Inter';
                this.ctx.fillStyle = '#ffffff';
                this.ctx.textAlign = 'center';
                this.ctx.textBaseline = 'middle';
                
                let text = '';
                if (l === 0) {
                    text = `x${n + 1}`;
                } else if (l === this.nodes.length - 1) {
                    text = 'y';
                } else {
                    text = `h${l}_${n + 1}`;
                }
                
                this.ctx.fillText(text, node.x, node.y);
            }
        }
        
        // 3. Draw Hover Tooltip if node is hovered
        if (this.hoveredNode) {
            const { layerIndex, nodeIndex } = this.hoveredNode;
            const node = this.nodes[layerIndex][nodeIndex];
            
            let name = '';
            let valText = '';
            let biasText = '';
            
            if (layerIndex === 0) {
                name = `Input ${nodeIndex + 1}`;
                if (this.activations && this.activations[0]) {
                    valText = `Input Value: ${this.activations[0][nodeIndex].toFixed(3)}`;
                }
            } else if (layerIndex === this.nodes.length - 1) {
                name = `Output Neuron`;
                if (this.activations && this.activations[layerIndex]) {
                    valText = `Output Prob: ${this.activations[layerIndex][nodeIndex].toFixed(4)}`;
                }
                if (this.biases && this.biases[layerIndex-1]) {
                    biasText = `Bias (b): ${this.biases[layerIndex-1][nodeIndex].toFixed(4)}`;
                }
            } else {
                name = `Hidden Neuron (Layer ${layerIndex}, Node ${nodeIndex + 1})`;
                if (this.activations && this.activations[layerIndex]) {
                    valText = `Activation (a): ${this.activations[layerIndex][nodeIndex].toFixed(4)}`;
                }
                if (this.biases && this.biases[layerIndex-1]) {
                    biasText = `Bias (b): ${this.biases[layerIndex-1][nodeIndex].toFixed(4)}`;
                }
            }
            
            // Draw Tooltip Box
            const tooltipW = 220;
            const tooltipH = biasText ? 65 : 45;
            const tooltipX = Math.min(width - tooltipW - 10, Math.max(10, node.x - tooltipW / 2));
            const tooltipY = Math.max(10, node.y - node.radius - tooltipH - 8);
            
            this.ctx.fillStyle = 'rgba(10, 10, 20, 0.95)';
            this.ctx.strokeStyle = 'rgba(99, 102, 241, 0.6)';
            this.ctx.lineWidth = 1.5;
            this.ctx.shadowBlur = 10;
            this.ctx.shadowColor = 'rgba(0, 0, 0, 0.5)';
            
            this.ctx.beginPath();
            this.ctx.roundRect(tooltipX, tooltipY, tooltipW, tooltipH, 6);
            this.ctx.fill();
            this.ctx.stroke();
            
            this.ctx.shadowBlur = 0;
            this.ctx.font = 'bold 11px Outfit';
            this.ctx.fillStyle = '#ffffff';
            this.ctx.textAlign = 'left';
            this.ctx.fillText(name, tooltipX + 12, tooltipY + 16);
            
            this.ctx.font = '500 10px Inter';
            this.ctx.fillStyle = 'rgba(255, 255, 255, 0.85)';
            if (valText) {
                this.ctx.fillText(valText, tooltipX + 12, tooltipY + 34);
            }
            if (biasText) {
                this.ctx.fillText(biasText, tooltipX + 12, tooltipY + 48);
            }
        }
    }
    
    animate() {
        this.pulseOffset += 1;
        this.draw();
        this.animationFrameId = requestAnimationFrame(() => this.animate());
    }
    
    destroy() {
        if (this.animationFrameId) {
            cancelAnimationFrame(this.animationFrameId);
        }
    }
    
    // Evaluate a forward pass locally on the frontend given input coords [x1, x2]
    // so we can visualize node activations dynamically when hovering on the graphs
    evaluateForwardPass(inputs) {
        if (!this.weights || !this.biases) return;
        
        let current = inputs;
        const activations = [current];
        
        // Custom activation helpers
        const sigmoid = (x) => 1 / (1 + Math.exp(-x));
        const tanh = (x) => Math.tanh(x);
        const relu = (x) => Math.max(0, x);
        
        const actFn = this.activationName === 'relu' ? relu 
                    : this.activationName === 'sigmoid' ? sigmoid 
                    : tanh;
                    
        for (let l = 0; l < this.weights.length; l++) {
            const layerW = this.weights[l]; // shape (inputs_l, neurons_l)
            const layerB = this.biases[l];   // shape (neurons_l)
            const nextActivations = [];
            
            // Check if final layer (output layer is always sigmoid for classification)
            const currentAct = (l === this.weights.length - 1) ? sigmoid : actFn;
            
            for (let j = 0; j < layerB.length; j++) {
                let sum = layerB[j];
                for (let i = 0; i < current.length; i++) {
                    sum += current[i] * layerW[i][j];
                }
                nextActivations.push(currentAct(sum));
            }
            current = nextActivations;
            activations.push(current);
        }
        
        this.setLiveActivations(activations);
    }
}

// Export to global scope
window.NetworkVisualizer = NetworkVisualizer;
