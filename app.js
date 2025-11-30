// Precision Farming ML - Frontend JavaScript

document.getElementById('farmForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    // Collect form data
    const formData = new FormData(e.target);
    const data = {};

    for (let [key, value] of formData.entries()) {
        // Convert numeric fields
        if (key !== 'region' && key !== 'crop_type' && key !== 'irrigation_type' && 
            key !== 'fertilizer_type' && key !== 'crop_disease_status') {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }

    // Show loading indicator
    document.getElementById('loadingIndicator').style.display = 'block';
    document.getElementById('results').style.display = 'none';

    try {
        // Make API request
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data)
        });

        const result = await response.json();

        if (result.success) {
            displayResults(result.predictions);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        alert('Failed to get predictions: ' + error.message);
    } finally {
        document.getElementById('loadingIndicator').style.display = 'none';
    }
});

function displayResults(predictions) {
    // Update result values
    document.getElementById('waterResult').textContent = 
        predictions.water_requirement_mm_per_day.toFixed(2);
    document.getElementById('fertilizerResult').textContent = 
        predictions.fertilizer_requirement_kg_per_week.toFixed(2);
    document.getElementById('yieldResult').textContent = 
        predictions.expected_yield_kg_per_hectare.toFixed(0);

    // Update recommendations
    document.getElementById('waterRec').textContent = 
        predictions.irrigation_recommendation;
    document.getElementById('fertilizerRec').textContent = 
        predictions.fertilizer_recommendation;

    // Update tips
    const tipsList = document.getElementById('tipsList');
    tipsList.innerHTML = '';
    predictions.yield_optimization_tips.forEach(tip => {
        const li = document.createElement('li');
        li.textContent = tip;
        tipsList.appendChild(li);
    });

    // Show results
    document.getElementById('results').style.display = 'block';
    document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
}

function resetForm() {
    document.getElementById('farmForm').reset();
    document.getElementById('results').style.display = 'none';
}

// Initialize with default values on page load
window.addEventListener('load', () => {
    console.log('Precision Farming ML - Frontend Initialized');
});
