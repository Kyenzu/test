    document.getElementById('predictionForm').addEventListener('submit', async function (e) {
        e.preventDefault();

        // Collect form data
        const inputData = {
            "Age": parseFloat(document.getElementById('age').value),
            "Tumor Size (cm)": parseFloat(document.getElementById('tumorSize').value),
            "Cost of Treatment (USD)": 0, // If you need this, please adjust accordingly
            "Economic Burden (Lost Workdays per Year)": 0, // Adjust as needed
            "Country": 0, // Adjust as needed
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

        const requestData = { inputs: [inputData] };

        try {
            // Sending request to the backend API
            const response = await fetch('http://localhost:8000/api/hfp_prediction', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            const result = await response.json();

            // Handle prediction result
            const prediction = result.prediction[0];
            const predictedClass = prediction["1"] > prediction["0"] ? "Yes" : "No";
            const riskPercentages = result.risk_percentages;

            // Format risk details
            const riskDetails = Object.entries(riskPercentages)
                .map(([key, value]) => `${key}: ${value.toFixed(2)}%`)
                .join('\n');

            // Build result message
            const fullResult = `
                <h4 class="alert-heading">Prediction Result</h4>
                <p><strong>Prediction:</strong> ${predictedClass}</p>
                <pre><strong>Risk Breakdown:</strong>\n${riskDetails}</pre>
            `;

            document.getElementById('result').innerHTML = `
                <div class="alert alert-info text-start" role="alert">
                    ${fullResult}
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
