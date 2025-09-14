
## 🚀 Project Overview: Saathvik879/INTERN

**INTERN** is a full-stack internship management platform designed to streamline the process of discovering, managing, and applying for internships. Built with a robust Python backend and a dynamic frontend, this project showcases a clean architectural separation and modern deployment practices.

### 🔧 Tech Stack Highlights
- **Backend**: FastAPI-powered Python services (`ultimate_backend.py`) for handling API requests and database interactions.
- **Frontend**: Firebase-hosted web interface for seamless user experience.
- **Database**: SQLite (`internships.db`, `internships_v4.db`) for storing internship listings and user data.
- **Deployment**: Firebase for frontend hosting, and Uvicorn for serving the backend locally or remotely.

---

## 🖥️ Running the Backend Locally

To launch the backend server:

1. Open a terminal.
2. Navigate to the project directory.
3. Run the following command:
   ```bash
   uvicorn ultimate_backend:app --reload --host 0.0.0.0 --port 8000
   ```

This will start the FastAPI server on port `8000`, accessible via `http://localhost:8000`.

---

## 🌐 Accessing the Frontend

For the easiest experience, skip the local setup and head straight to the **live deployment**:

👉 [intern25.web.app](https://intern25.web.app)

This hosted version is fully functional and ready for use.

---

## 🛠️ Running Frontend Locally (Optional)

If you'd prefer to run the frontend on your machine:

1. Make sure you have Node.js and npm installed.
2. Navigate to the `public` directory.
3. Run:
   ```bash
   npm install
   npm start
   ```

This will spin up the frontend locally, typically on `http://localhost:3000`.

---
