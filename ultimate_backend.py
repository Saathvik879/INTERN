from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import firebase_admin
from firebase_admin import auth, credentials
import uvicorn
import tempfile
import os
import json
import random
from datetime import datetime, timedelta
import uuid
from typing import List, Dict, Optional, Any
import asyncio
from pydantic import BaseModel
import logging

# Import your resume parsing functions
import re
import spacy
import fitz  # PyMuPDF
import docx2txt
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize spacy
try:
    nlp = spacy.load("en_core_web_sm")
except:
    logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

# Initialize Firebase Admin SDK
try:
    # Initialize Firebase Admin SDK (use your service account key)
    cred = credentials.Certificate("path/to/your/serviceAccountKey.json")  # Replace with your path
    firebase_admin.initialize_app(cred)
except Exception as e:
    logger.warning(f"Firebase initialization failed: {e}. Using mock authentication.")

app = FastAPI(title="AI Internship Platform Backend", version="1.0.0")
security = HTTPBearer()

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Your resume parsing functions (from parse_resume.py)
SKILLS_DB = set([
    "python", "c", "c++", "html", "css", "js", "javascript", "typescript",
    "data structure", "algorithm", "algorithms", "blockchain", "solidity",
    "machine learning", "deep learning", "tensorflow", "pytorch", "keras",
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "sql", "mysql", "postgresql", "mongodb", "redis", "firebase",
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
    "git", "github", "gitlab", "linux", "ubuntu", "windows",
    "android", "ios", "flutter", "react native", "swift", "kotlin",
    "java", "spring", "hibernate", "maven", "gradle",
    "figma", "sketch", "adobe", "photoshop", "illustrator",
    "unity", "unreal", "blender", "3d modeling",
    "data science", "data analysis", "pandas", "numpy", "matplotlib",
    "selenium", "cypress", "jest", "junit", "testing"
])

SECTION_HEADERS = {
    "objective": [
        "objective", "resume objective", "career objective",
        "summary", "professional summary", "profile"
    ],
    "education": ["education", "qualifications"],
    "projects": ["projects"],
    "skills": ["skills", "technical skills", "programming languages"],
    "certifications": ["certifications", "certificate", "certificates", "workshop", "course"],
    "accomplishments": ["accomplishments", "achievements", "awards"],
    "contact": ["contact", "contact details", "linkedin", "phone", "email"],
    "declaration": ["declaration", "statement"]
}

CERTIFICATION_KEYWORDS = ["certificate", "certification", "workshop", "course", "diploma"]

# Resume parsing functions
def extract_text_and_links_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    linkedin_urls = []
    for page in doc:
        text += page.get_text() + "\n"
        for link in page.get_links():
            uri = link.get("uri", None)
            if uri and "linkedin.com" in uri.lower():
                linkedin_urls.append(uri)
    doc.close()
    return text, linkedin_urls

def extract_text_from_docx(docx_path) -> str:
    return docx2txt.process(docx_path)

def segment_by_headers(text: str) -> dict:
    sections = {}
    current_section = "header"
    sections[current_section] = []
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    for line in lines:
        lowered = line.lower()
        found_header = False
        for section, headers in SECTION_HEADERS.items():
            if any(h in lowered for h in headers):
                current_section = section
                sections.setdefault(current_section, [])
                found_header = True
                break
        if not found_header:
            sections.setdefault(current_section, [])
            sections[current_section].append(line)
    
    for sec in sections:
        sections[sec] = clean_section_text("\n".join(sections[sec]))
    return sections

def clean_section_text(text: str) -> str:
    text = re.sub(r'[\u2022•\-\*]+', '', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def extract_name(header_text: str) -> str:
    lines = header_text.strip().splitlines()
    if nlp:
        for line in lines[:3]:
            doc = nlp(line)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text.strip()
    return lines[0].strip() if lines else "Unknown"

def extract_skills(text: str) -> list:
    found_skills = set()
    text_lower = text.lower()
    for skill in SKILLS_DB:
        if skill in text_lower:
            found_skills.add(skill.title())
    return list(found_skills)

def extract_location(text: str) -> str:
    if nlp:
        doc = nlp(text)
        locs = [ent.text for ent in doc.ents if ent.label_ == "GPE"]
        if locs:
            return locs[0]
    
    text_lower = text.lower()
    indian_cities = ["hyderabad", "bangalore", "mumbai", "delhi", "chennai", "pune", "kolkata", "ahmedabad"]
    for city in indian_cities:
        if city in text_lower:
            return city.title()
    
    if "india" in text_lower:
        return "India"
    return "Not specified"

def extract_email(text: str) -> str:
    match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', text)
    return match.group(0) if match else None

def extract_phone(text: str) -> str:
    match = re.search(r'\b\d{10,13}\b', text)
    return match.group(0) if match else None

def extract_certifications_from_sections(sections: dict) -> str:
    combined_text = "\n".join(sections.values()).lower()
    lines = combined_text.splitlines()
    cert_lines = []
    for line in lines:
        if any(keyword in line for keyword in CERTIFICATION_KEYWORDS):
            cert_lines.append(line.strip())
    return "\n".join(cert_lines)

def parse_resume_text(text: str, linkedin_urls: list) -> dict:
    sections = segment_by_headers(text)
    name = extract_name(sections.get("header", ""))
    skills = extract_skills(text)
    location = extract_location(sections.get("header", "") + " " + sections.get("education", ""))
    
    return {
        "name": name,
        "email": extract_email(text),
        "phone": extract_phone(text),
        "linkedin": linkedin_urls[0] if linkedin_urls else None,
        "location": location,
        "skills": skills,
        "objective": sections.get("objective", ""),
        "education": sections.get("education", ""),
        "projects": sections.get("projects", ""),
        "certifications": extract_certifications_from_sections(sections),
        "accomplishments": sections.get("accomplishments", ""),
        "contact": f"Email: {extract_email(text)}\nPhone: {extract_phone(text)}",
        "declaration": sections.get("declaration", "")
    }

def determine_experience_level(parsed_resume: dict) -> str:
    """Determine experience level based on resume content"""
    education = parsed_resume.get("education", "").lower()
    projects = parsed_resume.get("projects", "").lower()
    accomplishments = parsed_resume.get("accomplishments", "").lower()
    
    # Count indicators of experience
    experience_indicators = 0
    
    if "graduate" in education or "master" in education or "phd" in education:
        experience_indicators += 2
    
    if len(parsed_resume.get("skills", [])) > 8:
        experience_indicators += 1
    
    if "internship" in projects or "project" in projects:
        if projects.count("project") >= 3:
            experience_indicators += 2
        elif projects.count("project") >= 1:
            experience_indicators += 1
    
    if "award" in accomplishments or "achievement" in accomplishments:
        experience_indicators += 1
    
    if experience_indicators >= 4:
        return "experienced"
    elif experience_indicators >= 2:
        return "intermediate"
    else:
        return "entry"

def summarize_resume(parsed_resume: dict):
    """Enhanced resume summarization with better role detection"""
    roles = []
    skills_text = " ".join(parsed_resume.get("skills", [])).lower()
    projects_text = parsed_resume.get("projects", "").lower()
    education_text = parsed_resume.get("education", "").lower()
    
    # Data Science roles
    if any(keyword in skills_text for keyword in ["python", "data", "machine learning", "tensorflow", "pandas", "numpy"]):
        roles.append("Data Science")
    
    # Web Development roles
    if any(keyword in skills_text for keyword in ["html", "css", "javascript", "react", "angular", "vue"]):
        if any(keyword in skills_text for keyword in ["node.js", "express", "django", "flask"]):
            roles.append("Full Stack Developer")
        else:
            roles.append("Frontend Developer")
    
    # Backend Development
    if any(keyword in skills_text for keyword in ["node.js", "express", "django", "flask", "spring", "sql"]):
        if "Frontend Developer" not in roles:
            roles.append("Backend Developer")
    
    # Mobile Development
    if any(keyword in skills_text for keyword in ["android", "ios", "flutter", "react native", "swift", "kotlin"]):
        roles.append("Mobile Developer")
    
    # DevOps
    if any(keyword in skills_text for keyword in ["aws", "azure", "docker", "kubernetes", "jenkins"]):
        roles.append("DevOps Engineer")
    
    # UI/UX Design
    if any(keyword in skills_text for keyword in ["figma", "sketch", "adobe", "design", "ui", "ux"]):
        roles.append("UI/UX Designer")
    
    # Blockchain
    if any(keyword in skills_text for keyword in ["blockchain", "solidity", "smart contract"]):
        roles.append("Blockchain Developer")
    
    # Default fallback
    if not roles:
        if "computer" in education_text or "engineering" in education_text:
            roles = ["Software Developer"]
        else:
            roles = ["Software Developer"]
    
    # Convert to intern roles
    intern_roles = [f"{role} Intern" for role in roles[:3]]  # Limit to top 3 roles
    
    experience_level = determine_experience_level(parsed_resume)
    skills_count = len(parsed_resume.get("skills", []))
    
    summary = f"A {experience_level}-level candidate with {skills_count} technical skills, interested in roles such as {', '.join(intern_roles)}. "
    summary += f"Strong background in {', '.join(parsed_resume.get('skills', [])[:5])}."
    
    return summary, intern_roles, experience_level

# Accurate Internship Generation based on Resume
def generate_accurate_internships(roles: List[str], skills: List[str], location: str, experience_level: str) -> List[Dict]:
    """Generate accurate internships based on parsed resume data"""
    
    # Indian companies with realistic data
    companies_data = {
        # Tier 1 - Top MNCs
        "Google India": {"logo": "🔍", "rating": 4.6, "tier": 1, "locations": ["Bangalore", "Hyderabad", "Mumbai"]},
        "Microsoft India": {"logo": "🏢", "rating": 4.5, "tier": 1, "locations": ["Bangalore", "Hyderabad", "Chennai"]},
        "Amazon India": {"logo": "📦", "rating": 4.3, "tier": 1, "locations": ["Bangalore", "Chennai", "Mumbai"]},
        "Meta India": {"logo": "📘", "rating": 4.4, "tier": 1, "locations": ["Bangalore", "Mumbai"]},
        
        # Tier 2 - Established Startups/Companies
        "Flipkart": {"logo": "🛒", "rating": 4.2, "tier": 2, "locations": ["Bangalore", "Delhi NCR"]},
        "Zomato": {"logo": "🍔", "rating": 4.1, "tier": 2, "locations": ["Delhi NCR", "Bangalore"]},
        "Swiggy": {"logo": "🚚", "rating": 4.0, "tier": 2, "locations": ["Bangalore", "Mumbai"]},
        "Paytm": {"logo": "💰", "rating": 3.9, "tier": 2, "locations": ["Delhi NCR", "Bangalore"]},
        "Ola": {"logo": "🚗", "rating": 3.8, "tier": 2, "locations": ["Bangalore", "Mumbai"]},
        "PhonePe": {"logo": "📱", "rating": 4.2, "tier": 2, "locations": ["Bangalore", "Pune"]},
        
        # Tier 3 - Service Companies
        "TCS": {"logo": "💼", "rating": 3.8, "tier": 3, "locations": ["Mumbai", "Chennai", "Kolkata", "Pune"]},
        "Infosys": {"logo": "🖥️", "rating": 3.9, "tier": 3, "locations": ["Bangalore", "Mysore", "Chennai"]},
        "Wipro": {"logo": "⚡", "rating": 3.7, "tier": 3, "locations": ["Bangalore", "Hyderabad", "Chennai"]},
        "HCL Technologies": {"logo": "🔧", "rating": 3.6, "tier": 3, "locations": ["Delhi NCR", "Chennai", "Bangalore"]},
        "Accenture": {"logo": "🎯", "rating": 3.8, "tier": 3, "locations": ["Bangalore", "Mumbai", "Hyderabad"]},
        
        # Tier 4 - Growing Startups
        "Freshworks": {"logo": "🌱", "rating": 4.0, "tier": 4, "locations": ["Chennai", "Bangalore"]},
        "Zoho": {"logo": "📊", "rating": 4.1, "tier": 4, "locations": ["Chennai", "Austin"]},
        "Razorpay": {"logo": "💳", "rating": 4.3, "tier": 4, "locations": ["Bangalore", "Delhi NCR"]},
        "InMobi": {"logo": "📲", "rating": 3.9, "tier": 4, "locations": ["Bangalore", "Delhi NCR"]},
        "Myntra": {"logo": "👗", "rating": 3.8, "tier": 4, "locations": ["Bangalore", "Mumbai"]},
    }
    
    # Role-specific skill requirements
    role_skills = {
        "Data Science Intern": ["Python", "Machine Learning", "SQL", "Pandas", "NumPy"],
        "Frontend Developer Intern": ["HTML", "CSS", "JavaScript", "React", "Vue"],
        "Backend Developer Intern": ["Node.js", "Python", "SQL", "Express", "Django"],
        "Full Stack Developer Intern": ["JavaScript", "React", "Node.js", "MongoDB", "SQL"],
        "Mobile Developer Intern": ["Android", "iOS", "Flutter", "React Native", "Kotlin"],
        "DevOps Engineer Intern": ["AWS", "Docker", "Kubernetes", "Linux", "Jenkins"],
        "UI/UX Designer Intern": ["Figma", "Sketch", "Adobe", "Prototyping", "Design"],
        "Blockchain Developer Intern": ["Solidity", "Blockchain", "Smart Contracts", "Web3", "Ethereum"],
        "Software Developer Intern": ["Programming", "Git", "Algorithms", "Data Structures", "Testing"]
    }
    
    internships = []
    
    for role in roles:
        role_internships = []
        companies_list = list(companies_data.keys())
        random.shuffle(companies_list)
        
        # Generate 8-12 internships per role
        for i, company_name in enumerate(companies_list[:random.randint(8, 12)]):
            company = companies_data[company_name]
            
            # Calculate match score based on skills overlap
            required_skills = role_skills.get(role, ["Programming", "Problem Solving"])
            skill_overlap = len(set(skills) & set(required_skills))
            base_match = min(90, 60 + (skill_overlap * 5))
            
            # Adjust based on experience level
            if experience_level == "experienced":
                match_score = min(95, base_match + 5)
            elif experience_level == "intermediate":
                match_score = base_match
            else:
                match_score = max(75, base_match - 5)
            
            # Add some randomness
            match_score = min(95, max(75, match_score + random.randint(-3, 3)))
            
            # Determine location
            internship_location = location if location in company["locations"] else random.choice(company["locations"])
            
            # Calculate stipend based on company tier and experience (MAX ₹25,000)
            if company["tier"] == 1:  # Top MNCs
                if experience_level == "experienced":
                    stipend = random.randint(20000, 25000)
                elif experience_level == "intermediate":
                    stipend = random.randint(15000, 22000)
                else:
                    stipend = random.randint(12000, 18000)
            elif company["tier"] == 2:  # Established startups
                if experience_level == "experienced":
                    stipend = random.randint(15000, 22000)
                elif experience_level == "intermediate":
                    stipend = random.randint(12000, 18000)
                else:
                    stipend = random.randint(8000, 15000)
            elif company["tier"] == 3:  # Service companies
                if experience_level == "experienced":
                    stipend = random.randint(12000, 18000)
                elif experience_level == "intermediate":
                    stipend = random.randint(10000, 15000)
                else:
                    stipend = random.randint(6000, 12000)
            else:  # Growing startups
                if experience_level == "experienced":
                    stipend = random.randint(10000, 20000)
                elif experience_level == "intermediate":
                    stipend = random.randint(8000, 15000)
                else:
                    stipend = random.randint(5000, 12000)
            
            # Ensure max stipend is 25k
            stipend = min(stipend, 25000)
            
            internship = {
                "id": str(uuid.uuid4()),
                "title": role,
                "company": company_name,
                "companyLogo": company["logo"],
                "location": internship_location,
                "remote": random.choice([True, False]) if internship_location != "Remote" else True,
                "duration": random.choice(["3 months", "6 months", "4 months"]),
                "description": f"Join {company_name} as a {role}. Work on real-world projects with experienced mentors and contribute to cutting-edge technology solutions.",
                "requirements": required_skills,
                "skills": required_skills[:3],  # Top 3 skills for display
                "stipend": stipend,
                "stipendRange": f"₹{stipend:,} - ₹{min(stipend + 3000, 25000):,}",
                "matchScore": match_score,
                "featured": random.choice([True, False]) if company["tier"] <= 2 else False,
                "applicationCount": random.randint(20, 150),
                "companyRating": company["rating"],
                "postedDate": (datetime.now() - timedelta(days=random.randint(1, 14))).isoformat(),
                "applyBy": (datetime.now() + timedelta(days=random.randint(7, 30))).isoformat(),
                "experienceRequired": experience_level,
                "skillsMatched": skill_overlap,
                "totalSkillsRequired": len(required_skills)
            }
            
            role_internships.append(internship)
        
        # Sort by match score (highest first)
        role_internships.sort(key=lambda x: x["matchScore"], reverse=True)
        internships.extend(role_internships)
    
    # Remove duplicates and limit results
    seen = set()
    unique_internships = []
    for internship in internships:
        key = (internship["company"], internship["title"])
        if key not in seen and len(unique_internships) < 50:  # Limit to 50 total
            seen.add(key)
            unique_internships.append(internship)
    
    return unique_internships

# Data Models
class UserProfile(BaseModel):
    display_name: str
    bio: Optional[str] = None
    skills: List[str] = []
    experience_level: str = "entry"
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    phone: Optional[str] = None

class ChatMessage(BaseModel):
    message: str
    context: Optional[Dict] = {}

# Mock user database (replace with real database in production)
users_db = {}
user_profiles = {}

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify Firebase token or use mock authentication"""
    try:
        token = credentials.credentials
        # Try Firebase auth first
        decoded_token = auth.verify_id_token(token)
        return decoded_token
    except Exception as e:
        # Mock authentication for development
        if token == "mock_token":
            return {"uid": "mock_user", "email": "test@example.com", "name": "Test User"}
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials"
        )

@app.get("/")
async def root():
    return {"message": "AI Internship Platform Backend", "version": "1.0.0", "status": "running"}

@app.post("/api/auth/register")
async def register_user(user_data: dict, current_user: dict = Depends(verify_token)):
    """Register a new user"""
    user_id = current_user["uid"]
    users_db[user_id] = {
        "uid": user_id,
        "email": current_user.get("email"),
        "display_name": user_data.get("display_name"),
        "created_at": datetime.now().isoformat(),
        "last_login": datetime.now().isoformat()
    }
    
    # Initialize user profile
    user_profiles[user_id] = {
        "display_name": user_data.get("display_name"),
        "bio": "",
        "skills": [],
        "experience_level": "entry",
        "location": None,
        "linkedin_url": None,
        "github_url": None,
        "portfolio_url": None,
        "phone": None
    }
    
    return {"message": "User registered successfully", "user": users_db[user_id]}

@app.get("/api/profile")
async def get_profile(current_user: dict = Depends(verify_token)):
    """Get user profile"""
    user_id = current_user["uid"]
    profile = user_profiles.get(user_id, {})
    return {"profile": profile}

@app.put("/api/profile")
async def update_profile(profile: UserProfile, current_user: dict = Depends(verify_token)):
    """Update user profile"""
    user_id = current_user["uid"]
    user_profiles[user_id] = profile.dict()
    return {"message": "Profile updated successfully", "profile": user_profiles[user_id]}

@app.post("/api/ai/analyze-resume")
async def analyze_resume(file: UploadFile = File(...), current_user: dict = Depends(verify_token)):
    """Analyze uploaded resume using your parse_resume.py functions"""
    try:
        user_id = current_user["uid"]
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Parse resume based on file type
            if file.filename.lower().endswith('.pdf'):
                text, linkedin_urls = extract_text_and_links_from_pdf(tmp_file_path)
            elif file.filename.lower().endswith(('.docx', '.doc')):
                text = extract_text_from_docx(tmp_file_path)
                linkedin_urls = []
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
            
            # Parse resume
            parsed_resume = parse_resume_text(text, linkedin_urls)
            summary, roles, experience_level = summarize_resume(parsed_resume)
            
            # Generate accurate internships
            internships = generate_accurate_internships(
                roles=roles,
                skills=parsed_resume.get("skills", []),
                location=parsed_resume.get("location", "Bangalore"),
                experience_level=experience_level
            )
            
            # Update user profile with resume data
            user_profiles[user_id] = {
                **user_profiles.get(user_id, {}),
                "skills": parsed_resume.get("skills", []),
                "experience_level": experience_level,
                "location": parsed_resume.get("location"),
                "linkedin_url": parsed_resume.get("linkedin"),
                "phone": parsed_resume.get("phone")
            }
            
            result = {
                "parsed_resume": parsed_resume,
                "summary": summary,
                "roles": roles,
                "experience_level": experience_level,
                "recommended_internships": internships[:20],  # Return top 20
                "analysis_score": random.randint(80, 95),
                "skills_count": len(parsed_resume.get("skills", [])),
                "recommendations": [
                    "Add more technical projects to showcase your skills",
                    "Include quantifiable achievements in your experience",
                    "Optimize resume format for ATS compatibility",
                    "Consider adding relevant certifications"
                ]
            }
            
            return result
            
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
                
    except Exception as e:
        logger.error(f"Resume analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Resume analysis failed: {str(e)}")

@app.post("/api/ai/chat")
async def ai_chat(message_data: ChatMessage, current_user: dict = Depends(verify_token)):
    """AI chatbot responses based on context"""
    try:
        user_id = current_user["uid"]
        message = message_data.message.lower()
        context = message_data.context
        user_profile = user_profiles.get(user_id, {})
        
        # Context-aware responses
        if any(word in message for word in ["resume", "upload", "cv"]):
            if user_profile.get("skills"):
                response = f"Great! I've analyzed your resume and found {len(user_profile['skills'])} skills. You're well-suited for {', '.join(context.get('roles', ['multiple']))} roles. Check the Discover tab for personalized matches!"
            else:
                response = "Upload your resume using the orange button below the header! I'll analyze your skills and find perfect internship matches tailored to your profile. 📄✨"
        
        elif any(word in message for word in ["internship", "job", "opportunity"]):
            if user_profile.get("skills"):
                response = f"Based on your skills in {', '.join(user_profile['skills'][:3])}, I've found high-quality internships with stipends up to ₹25,000! Check the Discover section for matches with 85-95% compatibility. 🎯"
            else:
                response = "Upload your resume first and I'll find personalized internships with accurate salary ranges and perfect role matches! 🚀"
        
        elif any(word in message for word in ["salary", "stipend", "pay"]):
            response = "Our internships offer competitive stipends ranging from ₹5,000 to ₹25,000 based on company tier, your experience level, and skills match. Top companies like Google and Microsoft offer the highest stipends! 💰"
        
        elif any(word in message for word in ["profile", "settings"]):
            response = "You can update your profile and settings from the user menu (click your avatar). Adding more details helps me provide better recommendations and higher match scores! 👤⚙️"
        
        elif any(word in message for word in ["help", "guide"]):
            response = """I can help you with:
📄 Resume analysis & skill extraction
🎯 Finding accurate internship matches (75-95% compatibility)
💰 Salary insights (up to ₹25,000 stipends)
👤 Profile optimization for better matches
📊 Career guidance based on your skills
🚀 Application strategy and tips

What would you like to explore?"""
        
        elif any(word in message for word in ["match", "score", "compatibility"]):
            response = "Our AI calculates match scores based on your skills overlap with job requirements, experience level, and location preferences. Scores of 85%+ indicate excellent fits with high acceptance chances! 🎯"
        
        elif any(word in message for word in ["company", "companies"]):
            response = "I work with top companies across all tiers - from Google, Microsoft, Amazon (Tier 1) to Flipkart, Zomato (Tier 2) and TCS, Infosys (Tier 3). Each offers different stipend ranges and growth opportunities! 🏢"
        
        else:
            responses = [
                "That's interesting! I specialize in resume analysis and finding perfect internship matches. How can I help you advance your career? 🤖",
                "I'd love to assist you! My expertise includes accurate job matching, salary insights, and career guidance. What specific help do you need? 💪",
                "Great question! Upload your resume to unlock my full potential with personalized recommendations and high-match internships! 🌟",
                f"Hello{' ' + user_profile.get('display_name', '') if user_profile.get('display_name') else ''}! I'm here to help you find the perfect internship. What can I assist you with today? 🎯"
            ]
            response = random.choice(responses)
        
        return {"response": response}
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return {"response": "I'm experiencing technical difficulties. Please try again! 🤖"}

@app.get("/api/internships/discover")
async def discover_internships(
    role: Optional[str] = None,
    location: Optional[str] = None,
    experience: Optional[str] = None,
    limit: int = 20,
    current_user: dict = Depends(verify_token)
):
    """Get internships with filters"""
    try:
        user_id = current_user["uid"]
        user_profile = user_profiles.get(user_id, {})
        
        # Generate internships based on user profile or defaults
        if user_profile.get("skills"):
            # Use user's profile for accurate matching
            roles = ["Data Science Intern", "Frontend Developer Intern", "Backend Developer Intern"]  # Default roles
            internships = generate_accurate_internships(
                roles=roles,
                skills=user_profile.get("skills", []),
                location=user_profile.get("location", "Bangalore"),
                experience_level=user_profile.get("experience_level", "entry")
            )
        else:
            # Generate default internships
            default_roles = ["Software Developer Intern", "Frontend Developer Intern", "Data Science Intern"]
            internships = generate_accurate_internships(
                roles=default_roles,
                skills=["Programming", "Problem Solving"],
                location="Bangalore",
                experience_level="entry"
            )
        
        # Apply filters
        filtered_internships = internships
        
        if role:
            filtered_internships = [i for i in filtered_internships if role.lower() in i["title"].lower()]
        
        if location and location != "remote":
            filtered_internships = [i for i in filtered_internships if location.lower() in i["location"].lower()]
        elif location == "remote":
            filtered_internships = [i for i in filtered_internships if i["remote"]]
        
        if experience:
            filtered_internships = [i for i in filtered_internships if i["experienceRequired"] == experience]
        
        # Sort by match score and limit results
        filtered_internships.sort(key=lambda x: x["matchScore"], reverse=True)
        
        return {
            "internships": filtered_internships[:limit],
            "total": len(filtered_internships),
            "filters_applied": {
                "role": role,
                "location": location,
                "experience": experience
            }
        }
        
    except Exception as e:
        logger.error(f"Discover internships error: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch internships")

@app.get("/api/dashboard")
async def get_dashboard_data(current_user: dict = Depends(verify_token)):
    """Get dashboard statistics"""
    user_id = current_user["uid"]
    user_profile = user_profiles.get(user_id, {})
    
    # Generate realistic stats
    stats = {
        "applications_sent": random.randint(5, 25),
        "interviews_scheduled": random.randint(2, 8),
        "offers_received": random.randint(0, 4),
        "profile_views": random.randint(50, 200),
        "success_rate": random.randint(70, 95)
    }
    
    # Recent activities
    activities = [
        "Resume analyzed successfully",
        "New internship matches found",
        "Profile updated",
        "Applied to Software Developer Intern at TCS",
        "Received interview invitation from Infosys"
    ]
    
    return {
        "user_stats": stats,
        "recent_activities": activities[:3],
        "profile_completion": 85 if user_profile.get("skills") else 45
    }

@app.post("/api/internships/{internship_id}/apply")
async def apply_internship(internship_id: str, current_user: dict = Depends(verify_token)):
    """Apply to an internship"""
    user_id = current_user["uid"]
    
    # Mock application process
    application = {
        "id": str(uuid.uuid4()),
        "internship_id": internship_id,
        "user_id": user_id,
        "status": "applied",
        "applied_at": datetime.now().isoformat()
    }
    
    return {
        "message": "Application submitted successfully!",
        "application": application,
        "next_steps": [
            "Your application will be reviewed within 3-5 business days",
            "You'll receive an email confirmation shortly",
            "Keep checking your dashboard for updates"
        ]
    }

@app.get("/api/analytics")
async def get_analytics(current_user: dict = Depends(verify_token)):
    """Get user analytics"""
    return {
        "profile_views": random.randint(80, 200),
        "match_success_rate": random.randint(80, 95),
        "skills_compatibility": random.randint(85, 98),
        "career_progress": random.randint(70, 90),
        "application_response_rate": random.randint(60, 85)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
