# AI-Driven Railway Network Traffic Anomaly & Threat Detection

An AI-powered cybersecurity solution designed to secure critical railway infrastructure by detecting network intrusions and cyber threats in real-time. This system analyzes network traffic logs using Machine Learning algorithms to identify deviations from normal behavior, integrating threat intelligence to proactively defend vital transit systems.

## 📌 Project Overview

Modern railway networks rely heavily on interconnected digital systems for signaling, operations, and passenger services. This makes them potential targets for cyberattacks. 

This project implements an early-warning defense system that:
* Monitors and analyzes vast volumes of railway network traffic logs[cite: 5].
* Utilizes unsupervised and supervised Machine Learning models to baseline normal behavior and detect anomalies[cite: 5].
* Integrates Threat Intelligence Feeds to categorize anomalies into specific attack vectors (e.g., DDoS, unauthorized access)[cite: 5].
* Prevents operational disruptions to critical components like **Signaling Systems** and **Passenger Information Systems** before a breach occurs[cite: 6].

---

## 🚀 Key Features

* **Real-Time Anomaly Detection:** Continuous monitoring of network traffic logs to flag malicious or deviant patterns instantly[cite: 4, 5].
* **Machine Learning Pipeline:** Leverages robust algorithms like **Random Forest** and **Isolation Forest** for accurate classification and outlier detection[cite: 5].
* **Threat Categorization:** Matches detected anomalies with real-world threat intelligence to identify specific cyber threats, such as Distributed Denial of Service (DDoS) or unauthorized access attempts[cite: 5].
* **Proactive Digital Defense:** Focuses on early warning mechanisms to maintain the operational integrity and safety of national transportation networks[cite: 6, 7].

---

## 🛠️ Tech Stack & Architecture

* **Language:** Python
* **Machine Learning Libraries:** Scikit-learn (Random Forest, Isolation Forest), Pandas, NumPy
* **Data Source:** Network traffic logs (PCAP analysis / NetFlow logs) [cite: 5]
* **Threat Intel Integration:** Structured threat intelligence feeds [cite: 5]


---

## 📂 Repository Structure

```text
├── data/                  # Sample network traffic logs & threat intelligence feeds
├── notebooks/             # Jupyter notebooks for data preprocessing & EDA
├── src/
│   ├── preprocessing.py   # Log parsing and feature engineering
│   ├── models.py          # Random Forest & Isolation Forest implementations
│   └── threat_intel.py    # Threat intelligence matching logic
├── main.py                # Pipeline execution script
├── requirements.txt       # Project dependencies
└── README.md              # Project documentation
