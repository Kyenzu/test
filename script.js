document.getElementById('predictionForm').addEventListener('submit', async function (e) {
    e.preventDefault();

    const inputData = {
        "Age": parseFloat(document.getElementById('age').value),
        "Tumor Size (cm)": parseFloat(document.getElementById('tumorSize').value),
        "Cost of Treatment (USD)": 0,
        "Economic Burden (Lost Workdays per Year)": 0,
        "Country": 0,
        "Gender": document.getElementById('gender').value,
        "Tobacco Use": document.getElementById('tobaccoUse').value,
        "Alcohol Consumption": document.getElementById('alcoholConsumption').value,
        "HPV Infection": document.getElementById('hpvInfection').value,
        "Betel Quid Use": document.getElementById('betelQuidUse').value,
        "Chronic Sun Exposure": document.getElementById('sunExposure').value,
        "Poor Oral Hygiene": document.getElementById('oralHygiene').value,
        "Diet (Fruits & Vegetables Intake)": document.getElementById('diet').value,
        "Family History of Cancer": document.getElementById('familyHistory').value,
        "Compromised Immune System": document.getElementById('immuneSystem').value,
        "Oral Lesions": document.getElementById('oralLesions').value,
        "Unexplained Bleeding": document.getElementById('bleeding').value,
        "Difficulty Swallowing": document.getElementById('swallowing').value,
        "White or Red Patches in Mouth": document.getElementById('patches').value,
        "Treatment Type": document.getElementById('treatmentType').value,
        "Early Diagnosis": document.getElementById('earlyDiagnosis').value
    };

    try {
        const response = await fetch('http://localhost:8000/api/hfp_prediction', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ inputs: [inputData] })
        });

        const result = await response.json();
        const prediction = result.prediction[0];

        // Assuming prediction is in the form of { "0": 0, "1": 100 } or similar
        const predictedClass = prediction["1"] > prediction["0"] ? "Yes" : "No";

        document.getElementById('result').innerHTML = `
            <div class="alert alert-info text-start" role="alert">
                <h4 class="alert-heading">Prediction Result</h4>
                <p><strong>Prediction:</strong> ${predictedClass}</p>
            </div>
        `;
    } catch (error) {
        console.error('Error:', error);
        document.getElementById('result').innerHTML = `
            <div class="alert alert-danger" role="alert">
                An error occurred while making the prediction.
            </div>
        `;
    }
});
