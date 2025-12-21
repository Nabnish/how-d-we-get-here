

# 🚀 How-d-We-Get-Here

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.x-blue" alt="Python"></a>
  <a href="https://nodejs.org/"><img src="https://img.shields.io/badge/Node.js-16-green" alt="Node.js"></a>
</p>

<p align="center">
  Full-stack project demonstrating <b>routing, search, and data flow</b> from backend APIs to a frontend interface.
</p>

---

## ✨ Features

<ul>
  <li>🔍 Efficient search on uploaded datasets using indexes</li>
  <li>💻 Interactive frontend UI for visualization</li>
  <li>⚙️ Modular backend APIs for extensibility</li>
  <li>📁 Supports file uploads and dynamic data processing</li>
</ul>

---

## 📂 Project Structure

<table>
<tr>
<th>Folder/File</th><th>Description</th>
</tr>
<tr>
<td><b>Backend/</b></td><td>Python backend logic (APIs, search)</td>
</tr>
<tr>
<td><b>frontend/</b></td><td>Frontend UI (React/Node)</td>
</tr>
<tr>
<td><b>uploads/</b></td><td>Data files & search indexes</td>
</tr>
<tr>
<td><b>pubmed_faiss.index</b></td><td>Example index file</td>
</tr>
<tr>
<td><b>pubmed_meta.pkl</b></td><td>Example metadata</td>
</tr>
<tr>
<td><b>package-lock.json</b></td><td>Node dependency lock</td>
</tr>
<tr>
<td><b>README.md</b></td><td>This documentation</td>
</tr>
</table>

---

## ⚡ Getting Started

### 1️⃣ Prerequisites

<ul>
<li>Python 3.x</li>
<li>Node.js + npm</li>
<li>Git</li>
</ul>

### 2️⃣ Installation

```bash
git clone https://github.com/Nabnish/how-d-we-get-here.git
cd how-d-we-get-here
```

#### Backend

```bash
cd Backend
pip install -r requirements.txt
```

#### Frontend

```bash
cd ../frontend
npm install
```

---

### 3️⃣ Running the Project

#### Start Backend

```bash
cd Backend
python app.py
# or for FastAPI
uvicorn main:app --reload
```

#### Start Frontend

```bash
cd frontend
npm start
```

Open your browser at <a href="http://localhost:3000">[http://localhost:3000](http://localhost:3000)</a>

---

## 🛠️ Usage

1. Upload your dataset in `uploads/`
2. Run backend to process data and build search index
3. Use frontend to **query and explore** data interactively

---

## 🤝 Contributing

<ol>
<li>Fork the repo</li>
<li>Create a branch <code>feat/your-feature</code></li>
<li>Push & open a Pull Request</li>
</ol>

---




