# MemoryPal

**MemoryPal** — A lightweight, Python-based speech and text processing project with database integration. It helps you process, organize and revisit textual/speech data with NLP features.

> **Status:** WIP — Core features include speech/text preprocessing, NLTK integration, and database connectivity.

---

## 🚀 What is MemoryPal?

MemoryPal is a Python project for experimenting with speech and text processing. It provides utilities to preprocess text, manage warnings, download NLTK resources safely, and connect to a database for persistent storage.

Key ideas:

* Speech and text preprocessing
* Error-handled NLTK setup
* Database connection (logs show successful DB integration)
* Modular codebase for future extension

---

## ✨ Features

* Speech/text preprocessing utilities
* NLTK integration with safer SSL handling
* Database connectivity for storing processed data
* Configurable through `.env` and `config.toml`
* Logging for initialization and error handling

---

## 🧭 Tech stack

* **Language:** Python 3.10+
* **Libraries:** NLTK, JSON, Regex, Math, OS, Pathlib
* **Environment:** Virtualenv (`venv`)
* **Config:** `.env`, `config.toml`

> See `requirements.txt` for the full list of dependencies.

---

## 🛠️ Quick start (development)

> Make sure you have Python 3.10+ installed.

1. Clone the repo

```bash
git clone https://github.com/Ranjeetkumariiitbh/MemoryPal.git
cd MemoryPal
```

2. Create and activate virtual environment

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. Run the app (example with enhanced RAG script)

```bash
python enhanced_rag_app.py
```

Logs will confirm database connection and initialization.

---

## ⚙️ Environment variables

Example `.env` variables:

```
DB_URI=your_database_connection
DEBUG=True
```

---

## 📂 Project structure

```
MemoryPal/
│── enhanced_rag_app.py      # Main app entry point
│── speech_processor.py      # Speech/text processing module
│── requirements.txt         # Dependencies
│── config.toml              # Configurations
│── .env                     # Environment variables
│── venv/                    # Virtual environment
│── __pycache__/             # Python cache
```

---

## 🤝 Contributing

Contributions are welcome! Please fork the repo, create a new branch, and open a pull request.

---

## 📄 License

This project is licensed under the MIT License — see the `LICENSE` file for details.

---

## ✉️ Contact

Created by **Ranjeet Kumar** — final-year B.Tech Mechatronics student at IIIT Bhagalpur.
<img width="1913" height="1027" alt="image" src="https://github.com/user-attachments/assets/bb193873-e29a-47bd-a45c-60d6a5b45749" />
<img width="1919" height="985" alt="image" src="https://github.com/user-attachments/assets/e0f46255-3dc3-4a4c-873e-56fe68dd8986" />
<img width="1919" height="1004" alt="image" src="https://github.com/user-attachments/assets/f28e296f-1782-4f21-bc98-d59bda5da131" />
<img width="1919" height="1007" alt="image" src="https://github.com/user-attachments/assets/5210d7a1-42ba-4fa1-a062-400d599b76bc" />
<img width="1919" height="985" alt="image" src="https://github.com/user-attachments/assets/6a2b4afb-4c88-4b3c-ad57-759c633419eb" />




