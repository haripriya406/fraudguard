const form = document.querySelector("#prediction-form");
const result = document.querySelector("#prediction-result");

if (form && result) {
    form.addEventListener("submit", async (event) => {
        event.preventDefault();
        const button = form.querySelector("button[type='submit']");
        button.disabled = true;
        result.hidden = true;

        const values = Object.fromEntries(new FormData(form).entries());
        try {
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(values),
            });
            const data = await response.json();
            if (!response.ok) {
                throw new Error(data.error || "The transaction could not be analyzed.");
            }
            result.dataset.status = data.prediction;
            result.innerHTML = `<strong>${data.prediction === "FRAUD" ? "Potential fraud detected" : "Likely legitimate"}</strong>
                Fraud probability: ${data.fraud_probability}% &nbsp;·&nbsp; Legitimate probability: ${data.legitimate_probability}%`;
        } catch (error) {
            result.dataset.status = "error";
            result.textContent = error.message || "Unable to reach the analysis service.";
        } finally {
            result.hidden = false;
            button.disabled = false;
        }
    });
}

const navigationLinks = document.querySelectorAll(".nav-link");
navigationLinks.forEach((link) => {
    link.addEventListener("click", () => {
        navigationLinks.forEach((item) => item.classList.remove("active"));
        link.classList.add("active");
    });
});
