/**
 * Representation Power of MLPs - Main Controller
 * Handles SPA navigation, dynamic UI inputs, training orchestration, Plotly charts, and playback.
 */

document.addEventListener('DOMContentLoaded', () => {
    // ----------------------------------------------------------------------
    // STATE VARIABLES
    // ----------------------------------------------------------------------
    let appState = {
        activeTab: 'playground',
        dataset: 'xor',
        noise: 0.10,
        activation: 'relu',
        learningRate: 0.03,
        epochs: 150,
        layers: [4, 4], // Default: 2 layers of 4 neurons
        
        // Playback state
        trainingData: null, // Response payload from backend
        currentEpochIndex: 0,
        isPlaying: false,
        playbackSpeed: 200, // ms per frame
        playbackIntervalId: null,
        
        // Comparison Mode state
        compDataset: 'xor',
        compNoise: 0.10,
        compModelA: { layers: [2], activation: 'sigmoid', lr: 0.03 },
        compModelB: { layers: [8, 8, 4], activation: 'relu', lr: 0.03 }
    };

    // ----------------------------------------------------------------------
    // VISUALIZERS INIT
    // ----------------------------------------------------------------------
    const playgroundVisualizer = new NetworkVisualizer('network-canvas');
    playgroundVisualizer.setTopology(appState.layers, appState.activation);

    // ----------------------------------------------------------------------
    // DOM ELEMENTS
    // ----------------------------------------------------------------------
    const tabBtns = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    // Playground controls
    const datasetBtns = document.querySelectorAll('[data-dataset]:not(.comp-dataset)');
    const noiseSlider = document.getElementById('noise-slider');
    const noiseVal = document.getElementById('noise-val');
    const lrSelect = document.getElementById('lr-select');
    const activationSelect = document.getElementById('activation-select');
    const epochsSlider = document.getElementById('epochs-slider');
    const epochsVal = document.getElementById('epochs-val');
    const layersSlider = document.getElementById('layers-slider');
    const layersVal = document.getElementById('layers-val');
    const neuronsContainer = document.getElementById('neurons-sliders-container');
    const trainBtn = document.getElementById('train-btn');
    const statsPanel = document.getElementById('stats-panel');
    const statEpoch = document.getElementById('stat-epoch');
    const statLoss = document.getElementById('stat-loss');
    const statAcc = document.getElementById('stat-acc');
    const progressBar = document.getElementById('progress-bar');
    
    // Playback controls
    const playbackBar = document.getElementById('playback-bar');
    const playBtn = document.getElementById('playback-play');
    const prevBtn = document.getElementById('playback-prev');
    const nextBtn = document.getElementById('playback-next');
    const timelineSlider = document.getElementById('timeline-slider');
    const timelineCurrentEpoch = document.getElementById('current-timeline-epoch');
    const timelineTotalEpochs = document.getElementById('total-timeline-epochs');
    const speedSlider = document.getElementById('speed-slider');
    const speedLabel = document.getElementById('speed-label');
    const repLayerSelect = document.getElementById('rep-layer-select');
    
    // Comparison controls
    const compDatasetBtns = document.querySelectorAll('.comp-dataset');
    const compNoiseSlider = document.getElementById('comp-noise-slider');
    const compNoiseVal = document.getElementById('comp-noise-val');
    const compABtn = document.getElementById('compare-btn');
    
    // ----------------------------------------------------------------------
    // TAB NAVIGATION
    // ----------------------------------------------------------------------
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const targetTab = btn.dataset.tab;
            
            // Toggle active state in navbar
            tabBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            // Toggle section visibility
            tabContents.forEach(content => {
                content.classList.remove('active');
                if (content.id === `${targetTab}-sec`) {
                    content.classList.add('active');
                }
            });
            
            appState.activeTab = targetTab;
            
            // Resize visualizers when clicking back
            if (targetTab === 'playground') {
                playgroundVisualizer.resize();
            } else if (targetTab === 'comparison') {
                // Trigger resizing of comparison plotly charts if needed
                window.dispatchEvent(new Event('resize'));
            }
        });
    });

    // ----------------------------------------------------------------------
    // DYNAMIC CONTROL INITIALIZATION (PLAYGROUND)
    // ----------------------------------------------------------------------
    function renderNeuronSliders() {
        neuronsContainer.innerHTML = '';
        const count = parseInt(layersSlider.value);
        
        // Match state layers array length to selected count
        while (appState.layers.length < count) {
            appState.layers.push(4); // Default to 4 neurons
        }
        if (appState.layers.length > count) {
            appState.layers = appState.layers.slice(0, count);
        }
        
        for (let i = 0; i < count; i++) {
            const row = document.createElement('div');
            row.className = 'neuron-slider-row';
            row.innerHTML = `
                <div class="label-val-row">
                    <label>Layer ${i + 1} Neurons</label>
                    <span id="n-val-${i}" class="badge">${appState.layers[i]}</span>
                </div>
                <input type="range" class="range-slider neuron-slider" data-idx="${i}" min="1" max="10" step="1" value="${appState.layers[i]}">
            `;
            neuronsContainer.appendChild(row);
            
            const slider = row.querySelector('.neuron-slider');
            slider.addEventListener('input', (e) => {
                const idx = parseInt(e.target.dataset.idx);
                const val = parseInt(e.target.value);
                appState.layers[idx] = val;
                document.getElementById(`n-val-${idx}`).textContent = val;
                
                // Update network visualizer layout preview (weights/biases will reset)
                playgroundVisualizer.setTopology(appState.layers, appState.activation);
            });
        }
        
        // Update network visualizer layout
        playgroundVisualizer.setTopology(appState.layers, appState.activation);
        updateHiddenLayerDropdown();
    }
    
    // Dataset buttons
    datasetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            datasetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            appState.dataset = btn.dataset.dataset;
            fetchInitialDataset();
        });
    });
    
    // Sliders
    noiseSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        appState.noise = val;
        noiseVal.textContent = val.toFixed(2);
        fetchInitialDataset();
    });
    
    epochsSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        appState.epochs = val;
        epochsVal.textContent = val;
    });
    
    layersSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        layersVal.textContent = val;
        renderNeuronSliders();
    });
    
    lrSelect.addEventListener('change', (e) => {
        appState.learningRate = parseFloat(e.target.value);
    });
    
    activationSelect.addEventListener('change', (e) => {
        appState.activation = e.target.value;
        playgroundVisualizer.setTopology(appState.layers, appState.activation);
    });
    
    function updateHiddenLayerDropdown() {
        repLayerSelect.innerHTML = '';
        for (let i = 0; i < appState.layers.length; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.textContent = `Hidden Layer ${i + 1} (${appState.layers[i]}n)`;
            repLayerSelect.appendChild(opt);
        }
        repLayerSelect.value = appState.layers.length - 1; // default to last hidden layer
    }
    
    repLayerSelect.addEventListener('change', () => {
        if (appState.trainingData) {
            updateTimelineFrame(appState.currentEpochIndex);
        }
    });

    // ----------------------------------------------------------------------
    // INITIAL PLOTS CONFIGURATION & HELPERS
    // ----------------------------------------------------------------------
    const darkPlotlyLayout = {
        paper_bgcolor: 'rgba(0, 0, 0, 0)',
        plot_bgcolor: 'rgba(0, 0, 0, 0)',
        font: {
            family: 'Inter, sans-serif',
            color: '#cbd5e1'
        },
        margin: { l: 45, r: 45, t: 25, b: 40 },
        showlegend: false,
        xaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zerolinecolor: 'rgba(255, 255, 255, 0.1)',
            tickfont: { color: '#9ca3af', size: 10 },
            fixedrange: true
        },
        yaxis: {
            gridcolor: 'rgba(255, 255, 255, 0.05)',
            zerolinecolor: 'rgba(255, 255, 255, 0.1)',
            tickfont: { color: '#9ca3af', size: 10 },
            fixedrange: true
        }
    };

    function parseScatterPoints(x1, x2, y) {
        const class0_x1 = [];
        const class0_x2 = [];
        const class1_x1 = [];
        const class1_x2 = [];
        
        for (let i = 0; i < y.length; i++) {
            if (y[i] === 0) {
                class0_x1.push(x1[i]);
                class0_x2.push(x2[i]);
            } else {
                class1_x1.push(x1[i]);
                class1_x2.push(x2[i]);
            }
        }
        
        return { class0_x1, class0_x2, class1_x1, class1_x2 };
    }
    
    async function fetchInitialDataset() {
        // We can hit /api/train with epochs=1 to generate dataset details instantly
        // without running full training. This gives a beautiful reactive dataset view!
        try {
            const initData = await window.API.trainNetwork({
                dataset_name: appState.dataset,
                layers: appState.layers,
                activation: appState.activation,
                learning_rate: appState.learningRate,
                epochs: 1,
                noise: appState.noise
            });
            
            // Plot initial dataset
            plotDecisionBoundary(initData.dataset, null, initData.grid);
            
            // Reset plots & timeline
            appState.trainingData = null;
            playbackBar.classList.add('hidden');
            statsPanel.classList.add('hidden');
            playgroundVisualizer.setWeightsAndBiases(null, null);
            
            // Reset representation space plot
            Plotly.purge('hidden-representation-plot');
            Plotly.purge('analytics-plot');
            
        } catch (e) {
            console.error("Failed to load initial dataset", e);
        }
    }

    // ----------------------------------------------------------------------
    // 1. PLOT DECISION BOUNDARY
    // ----------------------------------------------------------------------
    function plotDecisionBoundary(dataset, boundaryPreds, gridCoords, containerId = 'decision-boundary-plot') {
        const { class0_x1, class0_x2, class1_x1, class1_x2 } = parseScatterPoints(dataset.x1, dataset.x2, dataset.y);
        
        const traces = [];
        
        // 1. Heatmap contour if decision boundary data exists
        if (boundaryPreds && gridCoords) {
            // Reconstruct 2D z matrix from flat array
            const z = [];
            const size = gridCoords.size;
            for (let i = 0; i < size; i++) {
                z.push(boundaryPreds.slice(i * size, (i + 1) * size));
            }
            
            traces.push({
                x: gridCoords.x,
                y: gridCoords.y,
                z: z,
                type: 'heatmap',
                hoverinfo: 'x+y+z',
                colorscale: [
                    [0.0, 'rgba(249, 115, 22, 0.42)'], // Orange for Class 0
                    [0.5, 'rgba(10, 10, 22, 0.1)'],    // Neutral grid blur
                    [1.0, 'rgba(99, 102, 241, 0.42)']  // Indigo for Class 1
                ],
                showscale: false,
                zmin: 0,
                zmax: 1
            });
        }
        
        // 2. Class 0 Points
        traces.push({
            x: class0_x1,
            y: class0_x2,
            mode: 'markers',
            name: 'Class 0',
            marker: {
                color: '#f97316',
                size: 8,
                line: { color: '#070715', width: 1.5 },
                shadowblur: 5,
                shadowcolor: '#f97316'
            },
            type: 'scatter'
        });
        
        // 3. Class 1 Points
        traces.push({
            x: class1_x1,
            y: class1_x2,
            mode: 'markers',
            name: 'Class 1',
            marker: {
                color: '#6366f1',
                size: 8,
                line: { color: '#070715', width: 1.5 },
                shadowblur: 5,
                shadowcolor: '#6366f1'
            },
            type: 'scatter'
        });
        
        const layout = {
            ...darkPlotlyLayout,
            xaxis: { ...darkPlotlyLayout.xaxis, range: [-2, 2] },
            yaxis: { ...darkPlotlyLayout.yaxis, range: [-2, 2] },
            hovermode: 'closest'
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });
        
        // Hook up plotly hover event to feed coordinates to the canvas visualizer!
        const plotDiv = document.getElementById(containerId);
        if (plotDiv && containerId === 'decision-boundary-plot') {
            plotDiv.on('plotly_hover', (data) => {
                if (data && data.points && data.points[0]) {
                    const pt = data.points[0];
                    // Evaluate forward pass only if we have weights loaded
                    playgroundVisualizer.evaluateForwardPass([pt.x, pt.y]);
                }
            });
        }
    }

    // ----------------------------------------------------------------------
    // 2. PLOT HIDDEN REPRESENTATIONS (PCA WARPING)
    // ----------------------------------------------------------------------
    function plotHiddenRepresentation(dataset, repCoords, layerIdx) {
        const layerReps = repCoords[layerIdx]; // Array of shape (200, 2)
        if (!layerReps) return;
        
        const x_coords = layerReps.map(pt => pt[0]);
        const y_coords = layerReps.map(pt => pt[1]);
        
        const { class0_x1: c0_x, class0_x2: c0_y, class1_x1: c1_x, class1_x2: c1_y } = parseScatterPoints(x_coords, y_coords, dataset.y);
        
        const traces = [
            {
                x: c0_x,
                y: c0_y,
                mode: 'markers',
                marker: {
                    color: '#f97316',
                    size: 7,
                    line: { color: '#070715', width: 1.2 }
                },
                type: 'scatter'
            },
            {
                x: c1_x,
                y: c1_y,
                mode: 'markers',
                marker: {
                    color: '#6366f1',
                    size: 7,
                    line: { color: '#070715', width: 1.2 }
                },
                type: 'scatter'
            }
        ];
        
        const layout = {
            ...darkPlotlyLayout,
            margin: { l: 30, r: 30, t: 15, b: 30 }
        };
        
        Plotly.newPlot('hidden-representation-plot', traces, layout, { responsive: true, displayModeBar: false });
    }

    // ----------------------------------------------------------------------
    // 3. PLOT LOSS & ACCURACY CURVES (DUAL Y-AXIS)
    // ----------------------------------------------------------------------
    function plotAnalytics(history) {
        const epochs = history.map(h => h.epoch);
        const loss = history.map(h => h.loss);
        const accuracy = history.map(h => h.accuracy);
        
        const traces = [
            {
                x: epochs,
                y: loss,
                name: 'Loss',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#f97316', width: 2 },
                yaxis: 'y'
            },
            {
                x: epochs,
                y: accuracy,
                name: 'Accuracy',
                type: 'scatter',
                mode: 'lines',
                line: { color: '#14b8a6', width: 2 },
                yaxis: 'y2'
            }
        ];
        
        const layout = {
            ...darkPlotlyLayout,
            margin: { l: 45, r: 45, t: 25, b: 35 },
            xaxis: {
                ...darkPlotlyLayout.xaxis,
                title: { text: 'Epoch', font: { size: 9 } }
            },
            yaxis: {
                title: { text: 'Loss', font: { size: 10, color: '#f97316' } },
                tickfont: { color: '#f97316' },
                gridcolor: 'rgba(255, 255, 255, 0.03)',
                fixedrange: true
            },
            yaxis2: {
                title: { text: 'Accuracy', font: { size: 10, color: '#14b8a6' } },
                tickfont: { color: '#14b8a6' },
                overlaying: 'y',
                side: 'right',
                range: [0, 1.05],
                gridcolor: 'rgba(255, 255, 255, 0.03)',
                fixedrange: true
            },
            shapes: [
                // Vertical slider line at current epoch
                {
                    type: 'line',
                    x0: 1,
                    y0: 0,
                    x1: 1,
                    y1: 1,
                    yref: 'paper',
                    line: {
                        color: 'rgba(255, 255, 255, 0.5)',
                        width: 1.5,
                        dash: 'dot'
                    }
                }
            ]
        };
        
        Plotly.newPlot('analytics-plot', traces, layout, { responsive: true, displayModeBar: false });
    }
    
    function updateAnalyticsIndicator(epoch) {
        const update = {
            shapes: [{
                type: 'line',
                x0: epoch,
                y0: 0,
                x1: epoch,
                y1: 1,
                yref: 'paper',
                line: {
                    color: 'rgba(255, 255, 255, 0.65)',
                    width: 1.5,
                    dash: 'dot'
                }
            }]
        };
        Plotly.relayout('analytics-plot', update);
    }

    // ----------------------------------------------------------------------
    // PLAYBACK CONTROLLER TIMELINE FUNCTIONS
    // ----------------------------------------------------------------------
    function updateTimelineFrame(index) {
        if (!appState.trainingData) return;
        
        appState.currentEpochIndex = index;
        const record = appState.trainingData.history[index];
        const totalRecords = appState.trainingData.history.length;
        
        // Update stats
        statEpoch.textContent = `${record.epoch} / ${appState.epochs}`;
        statLoss.textContent = record.loss.toFixed(4);
        statAcc.textContent = `${(record.accuracy * 100).toFixed(1)}%`;
        
        // Progress bar percentage
        progressBar.style.width = `${(record.epoch / appState.epochs) * 100}%`;
        
        // Update timeline range values
        timelineSlider.value = index + 1;
        timelineCurrentEpoch.textContent = record.epoch;
        
        // Redraw decision boundary & representations
        plotDecisionBoundary(appState.trainingData.dataset, record.decision_boundary, appState.trainingData.grid);
        
        const selectedLayerIdx = parseInt(repLayerSelect.value);
        plotHiddenRepresentation(appState.trainingData.dataset, record.hidden_representations, selectedLayerIdx);
        
        // Update vertical timeline indicator
        updateAnalyticsIndicator(record.epoch);
        
        // Feed weights/biases to canvas visualizer
        playgroundVisualizer.setWeightsAndBiases(record.weights, record.biases);
        
        // Initial forward pass evaluation at coordinate [0,0]
        playgroundVisualizer.evaluateForwardPass([0, 0]);
    }
    
    function playTimeline() {
        if (appState.isPlaying) return;
        
        appState.isPlaying = true;
        playBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
        playBtn.classList.add('play-active');
        
        appState.playbackIntervalId = setInterval(() => {
            let nextIndex = appState.currentEpochIndex + 1;
            const maxIdx = appState.trainingData.history.length - 1;
            
            if (nextIndex > maxIdx) {
                // Loop back or pause
                nextIndex = 0;
            }
            
            updateTimelineFrame(nextIndex);
        }, appState.playbackSpeed);
    }
    
    function pauseTimeline() {
        if (!appState.isPlaying) return;
        
        appState.isPlaying = false;
        playBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
        playBtn.classList.remove('play-active');
        clearInterval(appState.playbackIntervalId);
    }
    
    // Playback events
    playBtn.addEventListener('click', () => {
        if (appState.isPlaying) {
            pauseTimeline();
        } else {
            playTimeline();
        }
    });
    
    prevBtn.addEventListener('click', () => {
        pauseTimeline();
        let idx = appState.currentEpochIndex - 1;
        if (idx < 0) idx = appState.trainingData.history.length - 1;
        updateTimelineFrame(idx);
    });
    
    nextBtn.addEventListener('click', () => {
        pauseTimeline();
        let idx = appState.currentEpochIndex + 1;
        if (idx >= appState.trainingData.history.length) idx = 0;
        updateTimelineFrame(idx);
    });
    
    timelineSlider.addEventListener('input', (e) => {
        pauseTimeline();
        const frameIdx = parseInt(e.target.value) - 1;
        updateTimelineFrame(frameIdx);
    });
    
    speedSlider.addEventListener('input', (e) => {
        const val = parseInt(e.target.value);
        appState.playbackSpeed = val;
        speedLabel.textContent = `${val}ms`;
        
        // Restart interval with new speed if playing
        if (appState.isPlaying) {
            pauseTimeline();
            playTimeline();
        }
    });

    // ----------------------------------------------------------------------
    // API SUBMISSION (PLAYGROUND TRAINING)
    // ----------------------------------------------------------------------
    trainBtn.addEventListener('click', async () => {
        pauseTimeline();
        trainBtn.disabled = true;
        trainBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training...';
        statsPanel.classList.remove('hidden');
        
        try {
            const data = await window.API.trainNetwork({
                dataset_name: appState.dataset,
                layers: appState.layers,
                activation: appState.activation,
                learning_rate: appState.learningRate,
                epochs: appState.epochs,
                noise: appState.noise
            });
            
            appState.trainingData = data;
            
            // Setup timeline components
            timelineSlider.max = data.history.length;
            timelineTotalEpochs.textContent = appState.epochs;
            playbackBar.classList.remove('hidden');
            
            // Plot analytics curves
            plotAnalytics(data.history);
            
            // Jump to first frame and play!
            updateTimelineFrame(0);
            playTimeline();
            
        } catch (e) {
            alert(`Training Failed: ${e.message}`);
        } finally {
            trainBtn.disabled = false;
            trainBtn.innerHTML = '<i class="fa-solid fa-play"></i> Train Neural Network';
        }
    });

    // ----------------------------------------------------------------------
    // COMPARISON MODE FUNCTIONS
    // ----------------------------------------------------------------------
    compDatasetBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            compDatasetBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            appState.compDataset = btn.dataset.dataset;
        });
    });
    
    compNoiseSlider.addEventListener('input', (e) => {
        const val = parseFloat(e.target.value);
        appState.compNoise = val;
        compNoiseVal.textContent = val.toFixed(2);
    });
    
    // Calculates total parameter count (Weights + Biases)
    function calculateParameterCount(inputSize, layers, outputSize) {
        let total = 0;
        let prevSize = inputSize;
        const fullTopology = [...layers, outputSize];
        
        for (let i = 0; i < fullTopology.length; i++) {
            const neurons = fullTopology[i];
            const wCount = prevSize * neurons;
            const bCount = neurons;
            total += wCount + bCount;
            prevSize = neurons;
        }
        return total;
    }
    
    compABtn.addEventListener('click', async () => {
        compABtn.disabled = true;
        compABtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Training Both Models...';
        
        // Reset metrics elements
        document.getElementById('comp-acc-a').textContent = 'Training...';
        document.getElementById('comp-loss-a').textContent = 'Training...';
        document.getElementById('comp-acc-b').textContent = 'Training...';
        document.getElementById('comp-loss-b').textContent = 'Training...';
        
        // Reconstruct layer config from inputs
        const parseLayers = (id) => {
            const val = document.getElementById(id).value.trim();
            if (!val) return [4];
            return val.split(',').map(n => parseInt(n.trim())).filter(n => !isNaN(n) && n > 0);
        };
        
        const layersA = parseLayers('comp-a-neurons');
        const layersB = parseLayers('comp-b-neurons');
        
        const configA = {
            dataset_name: appState.compDataset,
            layers: layersA,
            activation: document.getElementById('comp-a-activation').value,
            learning_rate: parseFloat(document.getElementById('comp-a-lr').value),
            epochs: 180, // standardized epochs for fast comparison
            noise: appState.compNoise
        };
        
        const configB = {
            dataset_name: appState.compDataset,
            layers: layersB,
            activation: document.getElementById('comp-b-activation').value,
            learning_rate: parseFloat(document.getElementById('comp-b-lr').value),
            epochs: 180,
            noise: appState.compNoise
        };
        
        try {
            // Train in parallel
            const [resA, resB] = await Promise.all([
                window.API.trainNetwork(configA),
                window.API.trainNetwork(configB)
            ]);
            
            const finalRecordA = resA.history[resA.history.length - 1];
            const finalRecordB = resB.history[resB.history.length - 1];
            
            // Calculate parameter counts
            const paramsA = calculateParameterCount(2, layersA, 1);
            const paramsB = calculateParameterCount(2, layersB, 1);
            
            // Display metrics
            document.getElementById('comp-acc-a').textContent = `${(finalRecordA.accuracy * 100).toFixed(1)}% (${paramsA} params)`;
            document.getElementById('comp-loss-a').textContent = finalRecordA.loss.toFixed(4);
            
            document.getElementById('comp-acc-b').textContent = `${(finalRecordB.accuracy * 100).toFixed(1)}% (${paramsB} params)`;
            document.getElementById('comp-loss-b').textContent = finalRecordB.loss.toFixed(4);
            
            // Draw Decision Boundary (Model A & B side-by-side)
            plotDecisionBoundary(resA.dataset, finalRecordA.decision_boundary, resA.grid, 'comp-decision-a');
            plotDecisionBoundary(resB.dataset, finalRecordB.decision_boundary, resB.grid, 'comp-decision-b');
            
            // Draw last hidden layer representation space
            // Model A
            const lastHiddenLayerIdxA = resA.history[0].hidden_representations.length - 1;
            plotHiddenSpace(resA.dataset, finalRecordA.hidden_representations, lastHiddenLayerIdxA, 'comp-hidden-a');
            
            // Model B
            const lastHiddenLayerIdxB = resB.history[0].hidden_representations.length - 1;
            plotHiddenSpace(resB.dataset, finalRecordB.hidden_representations, lastHiddenLayerIdxB, 'comp-hidden-b');
            
        } catch (e) {
            alert(`Comparison Training Failed: ${e.message}`);
        } finally {
            compABtn.disabled = false;
            compABtn.innerHTML = '<i class="fa-solid fa-code-compare"></i> Train & Compare Models';
        }
    });
    
    // Separate helper for static comparison hidden space plotting
    function plotHiddenSpace(dataset, repCoords, layerIdx, containerId) {
        const layerReps = repCoords[layerIdx];
        if (!layerReps) return;
        
        const x_coords = layerReps.map(pt => pt[0]);
        const y_coords = layerReps.map(pt => pt[1]);
        
        const { class0_x1: c0_x, class0_x2: c0_y, class1_x1: c1_x, class1_x2: c1_y } = parseScatterPoints(x_coords, y_coords, dataset.y);
        
        const traces = [
            {
                x: c0_x, y: c0_y, mode: 'markers',
                marker: { color: '#f97316', size: 6, line: { color: '#070715', width: 1.0 } },
                type: 'scatter'
            },
            {
                x: c1_x, y: c1_y, mode: 'markers',
                marker: { color: '#6366f1', size: 6, line: { color: '#070715', width: 1.0 } },
                type: 'scatter'
            }
        ];
        
        const layout = {
            ...darkPlotlyLayout,
            margin: { l: 30, r: 30, t: 10, b: 30 }
        };
        
        Plotly.newPlot(containerId, traces, layout, { responsive: true, displayModeBar: false });
    }

    // ----------------------------------------------------------------------
    // EDUCATIONAL HUB INTERACTION
    // ----------------------------------------------------------------------
    const tocLinks = document.querySelectorAll('.toc-link');
    const articles = document.querySelectorAll('.edu-card');
    
    tocLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('href').substring(1);
            const targetArticle = document.getElementById(targetId);
            
            if (targetArticle) {
                targetArticle.scrollIntoView({ behavior: 'smooth' });
                
                tocLinks.forEach(l => l.classList.remove('active'));
                link.classList.add('active');
            }
        });
    });
    
    // Scroll spy for TOC links
    const scrollContainer = document.querySelector('.education-articles');
    if (scrollContainer) {
        window.addEventListener('scroll', () => {
            let currentActive = '';
            articles.forEach(article => {
                const rect = article.getBoundingClientRect();
                // If article header is near the top of the viewport
                if (rect.top <= 120) {
                    currentActive = article.id;
                }
            });
            
            if (currentActive) {
                tocLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${currentActive}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }

    // ----------------------------------------------------------------------
    // INITIALIZATION RUN
    // ----------------------------------------------------------------------
    renderNeuronSliders();
    fetchInitialDataset();
});
