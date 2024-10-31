<img src="cortex_logo.png" alt="Logo" width="200" />

# Corteχ
Corteχ (Core-tech) is designed to streamline patient selection for clinical trials by processing complex clinical trial criteria and matching patients based on their data. This application leverages natural language processing and a rules engine to assess patient suitability, providing scores and exclusion reasons based on specified inclusion and exclusion criteria.

Corteχ was designed in 24 hours for the Nucleate Pittburgh 2024 Biohackathon.

## Features
- **Natural Language Rule Parsing:** Input clinical trial criteria in natural language, and the app translates it into structured rules that the engine can understand.
- **Patient Scoring and Exclusion:** Each patient is assigned a score based on how well they meet the inclusion criteria, with mandatory exclusion rules applied as needed.
- **Logs and Records:** Exclusion reasons and scores are saved in a log folder for reference.
- **Customizable Criteria:** Supports age, gender, medication, and preexisting condition criteria, with a fallback option for unsupported conditions.

## Prerequisites
- **Python 3.7+**
- Required Python libraries:
  - `pandas`
  - `customtkinter`
  - `tkinter`
  - `anthropic` (for rule building through Claude API)
  - `PIL` (for image handling)
- **Anthropic API Key:** Store your key in a secret.txt file in the project root.

## Installation
1. Clone this repository:

```bash
git clone https://github.com/yourusername/clinical-trial-patient-selector.git
cd clinical-trial-patient-selector
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Anthropic API Key:

- Place your API key in a file named secret.txt in the project root:
```txt
YOUR_ANTHROPIC_API_KEY
```

## Usage
1. **Start the Application:** Run the main file:

```bash
python main.py
```
2. Select Patient Data File:
- Click **"Select Patient CSV File"** and choose a CSV file containing patient data.
- **Note:** The CSV file should contain columns including `age`, `gender`, `prescriptions`, and `icd9_codes`.

3. Enter Clinical Trial Criteria:
- Describe the inclusion and exclusion criteria for the trial in natural language. For example:
```txt
Patients aged 58-70 with no history of heart disease, who are currently not on insulin.
```
4. Run the Selection:
- Click **"Run"** to start the patient evaluation.
- The output will display patients who meet the criteria, sorted by their suitability score.
5. Review Results:
- Logs are saved in the `/logs` folder, including exclusion reasons and patient scores.
- The application will show the percentage of patients who met the criteria.

## File Structure
```plaintext
clinical-trial-patient-selector/
│
├── main.py                # Main application file
├── secret.txt             # API key file for Anthropic
├── requirements.txt       # Required Python packages
├── logs/                  # Folder for exclusion reasons and patient scores
├── images/
│   └── cortex_logo.png    # Logo for the application
└── README.md              # Project documentation
```
## Example Patient CSV Format
Your CSV file should contain the following columns for accurate parsing:

| subject_id | first_name | last_name | age | gender | prescriptions | icd9_codes |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | 
| 1234 | John |Doe | 65 | M | ["insulin", "calcium gluconate"] | ["401.9", "250.00"] |

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments
- Anthropic API for rule building.
- CustomTkinter for the modern Tkinter widgets.
- All contributors who helped with testing and development. (Speicifically, Myra Haider, Liv Toft, Juhi Gupta, and Cleo Chen)

## Contact
For questions or suggestions, please open an issue or reach out to the repository maintainer, @Jonpot.

