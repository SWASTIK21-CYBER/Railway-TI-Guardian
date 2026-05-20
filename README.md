# AI-Driven Railway Network Traffic Anomaly & Threat Detection

An AI-powered cybersecurity solution designed to secure critical railway infrastructure by detecting network intrusions and cyber threats in real-time. This system analyzes network traffic logs using Machine Learning algorithms to identify deviations from normal behavior, integrating threat intelligence to proactively defend vital transit systems.

## 📌 Project Overview

Modern railway networks rely heavily on interconnected digital systems for signaling, operations, and passenger services. This makes them potential targets for cyberattacks. 

This project implements an early-warning defense system that:
* Monitors and analyzes vast volumes of railway network traffic logs.
* Utilizes unsupervised and supervised Machine Learning models to baseline normal behavior and detect anomalies.
* Integrates Threat Intelligence Feeds to categorize anomalies into specific attack vectors (e.g., DDoS, unauthorized access).
* Prevents operational disruptions to critical components like **Signaling Systems** and **Passenger Information Systems** before a breach occurs.

---

## 🚀 Key Features

* **Real-Time Anomaly Detection:** Continuous monitoring of network traffic logs to flag malicious or deviant patterns instantly.
* **Machine Learning Pipeline:** Leverages robust algorithms like **Random Forest** and **Isolation Forest** for accurate classification and outlier detection.
* **Threat Categorization:** Matches detected anomalies with real-world threat intelligence to identify specific cyber threats, such as Distributed Denial of Service (DDoS) or unauthorized access attempts.
* **Proactive Digital Defense:** Focuses on early warning mechanisms to maintain the operational integrity and safety of national transportation networks.

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python
* **Machine Learning Libraries:** Scikit-learn (Random Forest, Isolation Forest), Pandas, NumPy
* **Data Source:** Network traffic logs (PCAP analysis / NetFlow logs) 
* **Threat Intel Integration:** Structured threat intelligence feeds 


---

## 📂 Repository Structure

```text
├── app/                      # Web application for UI dashboard
│   ├── static/               # CSS, JavaScript, and asset files
│   └── templates/            # HTML templates (Flask/Jinja)
├── assets/
│   └── screenshots/          # UI screenshots and project diagrams
├── config/                   # Configuration files (environment, constants)
├── data/                     # Project data store
│   ├── processed/            # Cleaned and engineered features for ML
│   └── raw/                  # Raw network traffic logs & threat intel feeds
├── logs/                     # System and application runtime logs
├── models/                   # Saved/trained ML models (pickle/joblib files)
├── notebooks/                # Jupyter notebooks for data analysis & EDA
├── src/                      # Source code for the core backend pipeline
├── tests/                    # Unit tests and validation scripts
└── venv/                     # Python virtual environment (hidden/ignored in production)
