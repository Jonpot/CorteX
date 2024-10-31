from typing import Any
import pandas as pd
import numpy as np
import json
import ast
import os
import anthropic

# Import the anthropic API key from secret.txt
os.environ['ANTHROPIC_API_KEY'] = open("secret.txt").read().strip()

class Patient:
    def __init__(self, patient_index: int, patient_data: pd.DataFrame):
        self.patient_index = patient_index
        self.patient_data = patient_data
        self.score = 0.0

class RuleBuilder:
    def __init__(self, rule_description: str):
        self.rule_description = rule_description

    def build_rule(self) -> dict[str, Any]:
        """
        Build a rule from a description.

        Returns:
            dict[str, Any]: The rule.
        """
        client = anthropic.Anthropic()
        system_prompt = """
            You are a top-of-the-line symbolic translator for clinical trials.
            You were built to solve the problem that, while patient data is well-defined
            and regularly structured, clinical trial definitions are not.

            You take natural-language clinical trial definitions that contain patient
            inclusion and exclusion criteria and convert them into a well-defined set of
            symbols that a program will be able to use to quickly parse well-formatted
            patient data files and assign a score to each trial based on the inclusion
            and exclusion criteria for the given trial.

            Your responses will always take the following format, without any extraneous
            prepended nor appended text:

            {
            "response": "rules",
            "inclusion_criterium": [
                {
                "rule": {
                        "type": "age",
                        "min": 58,
                        "max": 70
                    },
                    "weight": 1.0
                }
                },
                {
                ...
                }
            ]
            "exclusion_criterium": [
                {
                "rule": {
                        "type": "gender",
                        "gender": 1
                    }
                },
                {
                ...
                }
            ]
            }

            Specifications:
            - Every response must include a list of inclusion rules and exclusions rules
            - Both types of rules will have a list of criterium for the program to consider
            - The exclusion criterium will be considered to always be mandatory (1.0 weight)
            - The inclusion criterium can be more lenient, and you will assess how important the inclusion criteria is by assigning a weight from 0.0 to 1.0, which the program will consider when assigning a score to each potential patient.
            - Below you will see the strict dictionary you must stick to. These are the only criteria the engine knows how to verify at the current moment. However, if you feel like you cannot deliver a sufficiently good ruleset given the current dictionary, you may include the "other" rule type, described below.

            Symbol Dictionary:
            age: used to specify an age criteria for a patient.
            {
            “type”: “age”,
            “min”: <int> [years] (optional),
            “max”: <int> [years] (optional)
            }

            gender: used to specify a gender criteria for a patient.
            {
            “type”: “gender”,
            “gender”: <int> [0 for male, 1 for female, 2 for either] (2 is equivalent to not making this a rule)
            }

            medications: used to find whether a patient is prescribed certain medications.
            {
            “type”: “medications”,
            “medications”: <list[str]> [the lowercase names of specific medications] (these will be substring matched against the list of patient medications)
            }

            Example of medications:
            drug
            potassium chloride
            d5w
            sodium chloride
            ns
            furosemide
            insulin
            iso-osmotic dextrose
            5% dextrose
            sw
            magnesium sulfate
            morphine sulfate
            acetaminophen
            heparin
            calcium gluconate

            preexisting_conditions: used to find whether a patient has had prior diagnoses or conditions.
            {
            “type”: “preexisting_conditions”,
            “icd9_codes: <list[str]> [the ICD-9 codes describing the relevant conditions]
            }


            other: **Not to be used unless deemed strictly necessary.** Will inform the developers to prioritize work on adding capability to the engine to parse a rule of this format.
            {
            "type": "other"
            ... [whatever arguments you think are appropriate]
            }



            Notes:
            - Your goal is to be thorough and accurate. Clinical trials have been designed by medical professionals like yourself, and often certain criteria are only alluded to or implied. You should use the intuition from your training data to be as thorough and accurate as possible.
            - If for any reason you recieve a request that is not obviously a clinical trial definition, simply respond with:
            {
            "response": "error",
            "message": "Invalid clinical trial definition."
            }
        """
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            temperature=0.2,
            system=system_prompt,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": self.rule_description
                        }
                    ]
                }
            ]
        )

        rule = json.loads(message.content[0].text)
        return rule

class Engine:
    def __init__(self, rules: dict[str, Any], patient_csv_fp: str):
        self.rules = rules
        self.patient_csv_fp = patient_csv_fp

    def sort_patients(self) -> list[Patient]:
        """
        The engine takes in a set of rules and a CSV file containing patient data.

        It converts the CSV into a pandas DataFrame.

        It then iterates through the patients in the DataFrame and applies the rules to each patient to obtain a score.

        Finally, it sorts the patients by score and returns the patients in a list.
        """
        # Load the patient data from the CSV file
        patient_data = pd.read_csv(self.patient_csv_fp)

        # Create a list to store the patients
        patients = []

        # Iterate through the patients
        for index, row in patient_data.iterrows():
            patient = Patient(index, row)
            patients.append(patient)

        # Apply the appropriate rules to each patient
        for patient in patients:
            score = 0.0

            # Apply the inclusion criteria
            for inclusion_criterium in self.rules["inclusion_criterium"]:
                rule = inclusion_criterium["rule"]
                weight = inclusion_criterium["weight"]

                if rule["type"] == "age":
                    score += self.age(patient, min=rule["min"], max=rule["max"]) * weight
                elif rule["type"] == "gender":
                    score += self.gender(patient, gender=rule["gender"]) * weight
                elif rule["type"] == "medications":
                    score += self.medications(patient, medications=rule["medications"]) * weight
                elif rule["type"] == "preexisting_conditions":
                    score += self.preexisting_conditions(patient, study_icd9_codes=rule["icd9_codes"]) * weight

            patient.score = score
            # Apply the exclusion criteria (set score to 0 if any exclusion criteria are met)
            for exclusion_criterium in self.rules["exclusion_criterium"]:
                rule = exclusion_criterium["rule"]

                if rule["type"] == "age":
                    if self.age(patient, min=rule["min"], max=rule["max"]):
                        patient.score = 0.0
                        print(f'Patient {patient.patient_index} excluded due to age ({patient.patient_data["age"]}, needed {rule["min"]} to {rule["max"]})')
                        break
                elif rule["type"] == "gender":
                    if self.gender(patient, gender=rule["gender"]):
                        patient.score = 0.0
                        print(f'Patient {patient.patient_index} excluded due to gender ({patient.patient_data["gender"]}, needed to be {['M','F'][rule["gender"]]})')
                        break
                elif rule["type"] == "medications":
                    if self.medications(patient, medications=rule["medications"]) > 0.0:
                        patient.score = 0.0
                        print(f'Patient {patient.patient_index} excluded due to medications ({patient.patient_data["prescriptions"]}, but couldn\'t have {rule["medications"]})')
                        break
                elif rule["type"] == "preexisting_conditions":
                    if self.preexisting_conditions(patient, study_icd9_codes=rule["icd9_codes"]) > 0.0:
                        patient.score = 0.0
                        print(f'Patient {patient.patient_index} excluded due to preexisting conditions ({patient.patient_data["icd9_codes"]}, but couldn\'t have {rule["icd9_codes"]})')
                        break

        # Sort the patients by score
        patients.sort(key=lambda x: x.score, reverse=True)
        return patients
    
    def age(self, patient: Patient, *,
            min: int | None = None, max: int | None = None) -> float:
        """
        Used to specify an age criteria for a patient.

        Args:
            patient (Patient): The patient to evaluate.
            min (int): The minimum age (optional).
            max (int): The maximum age (optional).

        Returns:
            float: 1.0 if the patient's age is within the specified range, 0.0 otherwise.
        """
        age = patient.patient_data['age']
        if min is not None and age < min:
            return 0.0
        if max is not None and age > max:
            return 0.0
        return 1.0
    
    def gender(self, patient: Patient, gender: int) -> float:
        """
        Used to specify a gender criteria for a patient.

        Args:
            patient_data (Patient): The patient to evaluate
            gender (int): 0 for male, 1 for female, 2 for either

        Returns:
            float: 1.0 if the patient is of the specified gender, 0.0 otherwise.
        """
        if gender == 0:
            if patient.patient_data['gender'] == 'M':
                return 1.0

        elif gender == 1:
            if patient.patient_data['gender'] == 'F':
                return 1.0

        elif gender == 2:
            return 1.0

        else:
            return 0.0
        
    def medications(self, patient: Patient, medications: list[str] = None) -> float:
        """
        Used to find whether a patient is prescribed certain medications.

        Args:
            patient_data (Patient): The patient to evaluate
            medications (list[str]): List of medications

        Returns:
            float: between 1.0 and 0.0 for the percent the patient's medications that match the listed medications
        """
        # Columns to check in patient data
        columns = ['prescriptions', 'prescriptions_poe', 'prescriptions_generic']
        matches = 0

        for med in medications:
            # convert medication name to all lowercase
            med = med.lower().strip()

            for col in columns:
                for patient_med in patient.patient_data[col]:
                    if  med in patient_med.lower().strip():
                        matches += 1

        total = len(medications)

        return matches/total
    
    def preexisting_conditions(self, patient: Patient, study_icd9_codes: list[str]) -> float:
        """
        Used to find whether a patient has had prior diagnoses or conditions.

        Args:
            patient_data (Patient): The patient to evaluate
            study_icd9_codes (list[str]): List of ICD-9 codes

        Returns:
            float: between 1.0 and 0.0 for the percent the patient's prior conditions that match the listed ICD-9 codes
        """

        patient_icd9_codes = patient.patient_data['icd9_codes']

        # Standardize ICD-9 codes for clean string matching
        patient_icd9_codes = patient_icd9_codes.strip().lower()
        study_icd9_codes = [x.strip().lower() for x in study_icd9_codes]

        # Convert ICD-9 diagnostic codes to sets
        patient_icd9_codes = set(ast.literal_eval(patient_icd9_codes))
        study_icd9_codes = set(study_icd9_codes)

        # Count how many ICD-9 codes the patient matches to the study
        set_diff = patient_icd9_codes.intersection(study_icd9_codes)

        # Return a binary score of 1 or 0 if patient matches all or no criteria
        # Return a raw score for ranking
        prior_condition_score = len(set_diff) / len(study_icd9_codes)

        return prior_condition_score
    

# Define the rules
rule_description = """
                    ID: NCT00744835
                    "Inclusion Criteria:\n\n1. The age of ≥ 18 years of age and ≤ 70 years of age, gender is not limited;\n2. Confirmed or suspected bacterial infection (refer to Appendix 5);\n3. Infection-related organ failure does not exceed 24 hours; organ failure is defined as circulation, (SOFA) ≥ 3 points in at least one organ or system of the respiratory, kidney, liver, coagulation and central nervous system;\n4. The time interval between the selection of the test drug and the test drug is not more than 8 hours;\n5. Infertility test for women of childbearing age;\n6. Childbearing age within six months without child care plan and agreed to take effective measures during the study of contraception;\n7. Patients or legal representatives signed informed consent.

                    \n\nExclusion Criteria:\n\n1. Pregnant or lactating women, or unable to take effective measures of contraception;\n2. Patients are expected to live less than 28 days due to basic diseases, such as poor control of malignant tumor, cardiac arrest in 30 days, and end-stage lung disease.\n3. The patient has the following chronic organ dysfunction or immunosuppression (based on the chronic health scoring assessment of the APACHE II score) : a) heart: New York heart association cardiac function IV; B) breathing: chronic obstructive, obstructive, or vascular lung disease can lead to severe restrictions on activities, i.e. the inability to go upstairs or to do housework; Or clear chronic hypoxia, CO2 retention, secondary real erythrocyte, severe pulmonary hypertension (\\> 40 mmHg) or respiratory muscle dependence; C) kidneys: receiving long-term dialysis; D) liver: liver cirrhosis confirmed by biopsy and clear portal hypertension; The upper digestive tract hemorrhage caused by portal hypertension; Or previous liver failure/hepatic encephalopathy/hepatic coma; E) of immune function: accept the treatment of impact resistance to infection, such as immune suppression therapy, chemotherapy, radiation therapy, or for a long time the recent use of high doses of hormones), or sickness impact resistance to infection, such as leukemia, lymphoma and AIDS);\n4. Solid organ or bone marrow transplantation;\n5. Plant survival status;\n6. The following conditions occurred within 4 weeks prior to infection: a) acute pulmonary embolism; B) transfusion response; C) acute coronary syndrome;\n7. Confirmed or highly suspected of acute infectious diseases such as viral hepatitis activity and active tuberculosis;\n8. Patients with sinus bradycardia (less than 60 per minute);\n9. Severe anemia (hemoglobin \\< 7.0 g/dL);\n10. Uncontrolled bleeding in the past 24 hours;\n11. Large area burns or chemical burns (III degree burns area \\> 30% BSA);\n12. The average arterial pressure was \\< 65 mmHg after adequate liquid resuscitation and vasoactive drug therapy.\n13. Acute myeloid hematopoiesis was characterized by a lack of severe granulocytes (ANC \\< 500 / mm3), or severe thrombocytopenia (\\< 20,000 / mm3);\n14. Allergic to the active ingredient or its auxiliary materials;\n15. The medication patients are using may severely affect the metabolism of the drug;\n16. Patients and (or) legal representatives have signed an unresuscitation (DNR), or decided to withdraw life support (withdraw) or restrict life support for the intensity (of the patient) and sign the informed consent form;\n17. Participated in clinical intervention test in 3 months;\n18. The subject is a researcher or his immediate family member, or may have improper informed consent;\n19. The attending physician considers it inappropriate for the patient to participate in this test.",
                    """

rule_builder = RuleBuilder(rule_description)
rules = rule_builder.build_rule()

# Create an instance of the engine
engine = Engine(rules, "bigPatientData.csv")

# Sort the patients
patients = engine.sort_patients()

# Print the patients
for patient in patients:
    if patient.score > 0:
        print(f'{patient.patient_data['first_name']} {patient.patient_data['last_name']}, ID: {patient.patient_data['subject_id']}. Score: {patient.score}')

# Print percentage of patients that meet the inclusion criteria
num_patients = len(patients)
num_included_patients = sum(patient.score > 0 for patient in patients)
percentage_included = num_included_patients / num_patients * 100
print(f"Percentage of patients that meet the inclusion criteria: {percentage_included}%")
print(f'Percentage of patients that were excluded: {100 - percentage_included}%')