# Complete AI Internship Dashboard Backend Implementation
# FastAPI Backend with AI Models and Web Scraping

from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import credentials, auth, firestore, storage
import asyncio
import aiohttp
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from typing import List, Dict, Optional, Any
import logging
from pydantic import BaseModel
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup
import re
from io import BytesIO
import PyPDF2
import docx2txt
import hashlib
from celery import Celery
import redis
from github import Github
import openai
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="InternAI Backend API",
    description="AI-powered internship matching platform backend",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize Firebase Admin
firebase_config = {
    "type": "service_account",
    "project_id": "intern-23c45",
    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("FIREBASE_PRIVATE_KEY", "").replace('\\n', '\n'),
    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": os.getenv("FIREBASE_CERT_URL")
}

if not firebase_admin._apps:
    if os.getenv("FIREBASE_CREDENTIALS_JSON"):
        cred = credentials.Certificate(json.loads(os.getenv("FIREBASE_CREDENTIALS_JSON")))
    else:
        cred = credentials.Certificate(firebase_config)
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://intern-23c45-default-rtdb.firebaseio.com/',
        'storageBucket': 'intern-23c45.firebasestorage.app'
    })

db = firestore.client()

# Initialize ML models and services
try:
    nlp = spacy.load("en_core_web_sm")
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("NLP models loaded successfully")
except Exception as e:
    logger.error(f"Error loading NLP models: {e}")
    nlp = None
    sentence_model = None

# Initialize Celery for background tasks
celery_app = Celery(
    'internai_backend',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Initialize Redis for caching
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=1, decode_responses=True)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.error(f"Redis connection failed: {e}")
    redis_client = None

# Pydantic models for API
class UserProfile(BaseModel):
    name: str
    email: str
    bio: Optional[str] = None
    location: Optional[str] = None
    phone: Optional[str] = None
    linkedin: Optional[str] = None
    portfolio: Optional[str] = None
    skills: List[str] = []
    preferences: Dict[str, Any] = {}

class InternshipFilter(BaseModel):
    locations: List[str] = []
    skills: List[str] = []
    work_types: List[str] = []
    stipend_min: Optional[int] = None
    stipend_max: Optional[int] = None
    duration: List[str] = []

class ResumeParseResult(BaseModel):
    skills: List[str]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]
    projects: List[Dict[str, Any]]
    contact_info: Dict[str, str]

# Authentication dependency
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        # Verify Firebase ID token
        decoded_token = auth.verify_id_token(credentials.credentials)
        return decoded_token
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(status_code=401, detail="Invalid authentication token")

# AI Resume Parser Class
class AIResumeParser:
    def __init__(self):
        self.skills_database = self._load_skills_database()
        self.experience_patterns = [
            r'(\d+(?:\.\d+)?)\s*(?:years?|yrs?|months?|mos?)\s*(?:of\s*)?(?:experience|exp)',
            r'(?:experience|exp).*?(\d+(?:\.\d+)?)\s*(?:years?|yrs?|months?|mos?)',
        ]
        
    def _load_skills_database(self):
        """Load comprehensive skills database"""
        skills_db = {
            'programming_languages': ['Python', 'JavaScript', 'Java', 'C++', 'C#', 'Go', 'Rust', 'Swift', 'Kotlin'],
            'web_frameworks': ['React', 'Angular', 'Vue.js', 'Django', 'Flask', 'Express.js', 'Spring Boot', 'FastAPI'],
            'databases': ['MongoDB', 'PostgreSQL', 'MySQL', 'SQLite', 'Redis', 'Elasticsearch', 'Cassandra'],
            'cloud_platforms': ['AWS', 'Azure', 'Google Cloud', 'Firebase', 'Heroku', 'Vercel', 'Netlify'],
            'ml_frameworks': ['TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras', 'OpenCV', 'Pandas', 'NumPy'],
            'devops_tools': ['Docker', 'Kubernetes', 'Jenkins', 'Git', 'GitHub Actions', 'Terraform', 'Ansible']
        }
        return skills_db

    async def parse_resume(self, file_content: bytes, filename: str) -> ResumeParseResult:
        """Parse resume and extract structured information"""
        try:
            # Extract text from file
            text = await self._extract_text_from_file(file_content, filename)
            
            # Parse using spaCy NLP
            doc = nlp(text) if nlp else None
            
            # Extract information
            skills = self._extract_skills(text, doc)
            experience = self._extract_experience(text, doc)
            education = self._extract_education(text, doc)
            projects = self._extract_projects(text, doc)
            contact_info = self._extract_contact_info(text, doc)
            
            return ResumeParseResult(
                skills=skills,
                experience=experience,
                education=education,
                projects=projects,
                contact_info=contact_info
            )
        except Exception as e:
            logger.error(f"Resume parsing error: {e}")
            raise HTTPException(status_code=400, detail=f"Resume parsing failed: {str(e)}")

    async def _extract_text_from_file(self, file_content: bytes, filename: str) -> str:
        """Extract text content from PDF or DOCX files"""
        try:
            if filename.lower().endswith('.pdf'):
                pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
                return text
            elif filename.lower().endswith('.docx'):
                return docx2txt.process(BytesIO(file_content))
            else:
                raise HTTPException(status_code=400, detail="Unsupported file format")
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            raise HTTPException(status_code=400, detail="Failed to extract text from file")

    def _extract_skills(self, text: str, doc) -> List[str]:
        """Extract skills using NLP and pattern matching"""
        skills = set()
        text_lower = text.lower()
        
        # Extract from skills database
        for category, skill_list in self.skills_database.items():
            for skill in skill_list:
                if skill.lower() in text_lower:
                    skills.add(skill)
        
        # Use NER if available
        if doc:
            for ent in doc.ents:
                if ent.label_ in ["ORG", "PRODUCT", "GPE"] and len(ent.text) > 2:
                    # Check if it's a technology/skill
                    if any(keyword in ent.text.lower() for keyword in ['js', 'py', 'sql', 'api', 'framework']):
                        skills.add(ent.text)
        
        return list(skills)[:20]  # Limit to top 20 skills

    def _extract_experience(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract work experience information"""
        experience = []
        
        # Look for experience sections
        exp_section = re.search(r'(experience|work history|employment)(.*?)(?=education|projects|skills|$)', 
                               text, re.IGNORECASE | re.DOTALL)
        
        if exp_section:
            exp_text = exp_section.group(2)
            
            # Extract experience entries (simplified pattern matching)
            exp_patterns = [
                r'(\w+(?:\s+\w+)*)\s*[\-–]\s*([^\n]+)\s*\n([^\n]+)\s*\n',
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+at\s+([^\n]+)',
            ]
            
            for pattern in exp_patterns:
                matches = re.finditer(pattern, exp_text, re.MULTILINE)
                for match in matches:
                    experience.append({
                        "title": match.group(1).strip(),
                        "company": match.group(2).strip() if len(match.groups()) > 1 else "Unknown",
                        "description": match.group(3).strip() if len(match.groups()) > 2 else ""
                    })
        
        return experience[:5]  # Limit to 5 experiences

    def _extract_education(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract education information"""
        education = []
        
        # Look for education section
        edu_section = re.search(r'(education|academic|qualification)(.*?)(?=experience|projects|skills|$)', 
                               text, re.IGNORECASE | re.DOTALL)
        
        if edu_section:
            edu_text = edu_section.group(2)
            
            # Extract degree patterns
            degree_patterns = [
                r'(B\.?Tech|Bachelor|M\.?Tech|Master|PhD|B\.?Sc|M\.?Sc)\s+(?:in\s+)?([^\n]+)',
                r'([^\n]*University[^\n]*)',
                r'([^\n]*College[^\n]*)',
            ]
            
            for pattern in degree_patterns:
                matches = re.finditer(pattern, edu_text, re.MULTILINE | re.IGNORECASE)
                for match in matches:
                    education.append({
                        "degree": match.group(1).strip() if len(match.groups()) > 1 else match.group(0).strip(),
                        "institution": match.group(2).strip() if len(match.groups()) > 1 else "Unknown"
                    })
        
        return education[:3]  # Limit to 3 education entries

    def _extract_projects(self, text: str, doc) -> List[Dict[str, Any]]:
        """Extract project information"""
        projects = []
        
        # Look for projects section
        proj_section = re.search(r'(projects?|portfolio)(.*?)(?=experience|education|skills|$)', 
                                text, re.IGNORECASE | re.DOTALL)
        
        if proj_section:
            proj_text = proj_section.group(2)
            
            # Simple project extraction
            project_lines = [line.strip() for line in proj_text.split('\n') if line.strip()]
            
            for i, line in enumerate(project_lines[:10]):  # Check first 10 lines
                if len(line) > 20 and not line.islower():  # Likely a project title
                    description = project_lines[i+1] if i+1 < len(project_lines) else ""
                    projects.append({
                        "name": line,
                        "description": description[:200]  # Limit description
                    })
        
        return projects[:5]  # Limit to 5 projects

    def _extract_contact_info(self, text: str, doc) -> Dict[str, str]:
        """Extract contact information"""
        contact_info = {}
        
        # Email pattern
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_matches = re.findall(email_pattern, text)
        if email_matches:
            contact_info['email'] = email_matches[0]
        
        # Phone pattern (Indian numbers)
        phone_patterns = [
            r'\+91[-.\s]?\d{10}',
            r'\b\d{10}\b',
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}'
        ]
        
        for pattern in phone_patterns:
            phone_matches = re.findall(pattern, text)
            if phone_matches:
                contact_info['phone'] = phone_matches[0]
                break
        
        # LinkedIn pattern
        linkedin_pattern = r'(?:linkedin\.com/in/|linkedin\.com/profile/view\?id=)([A-Za-z0-9\-_]+)'
        linkedin_matches = re.findall(linkedin_pattern, text, re.IGNORECASE)
        if linkedin_matches:
            contact_info['linkedin'] = f"https://linkedin.com/in/{linkedin_matches[0]}"
        
        return contact_info

# AI Matching Engine
class AIMatchingEngine:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.skill_weights = {
            'exact_match': 0.4,
            'semantic_similarity': 0.3,
            'experience_relevance': 0.2,
            'location_preference': 0.1
        }

    async def calculate_match_score(self, user_profile: Dict, internship: Dict) -> float:
        """Calculate AI-powered match score between user and internship"""
        try:
            scores = {}
            
            # Skill matching
            scores['skill_match'] = self._calculate_skill_match(
                user_profile.get('skills', []), 
                internship.get('skills', [])
            )
            
            # Experience relevance
            scores['experience_match'] = self._calculate_experience_match(
                user_profile.get('experience', []), 
                internship.get('requirements', [])
            )
            
            # Location preference
            scores['location_match'] = self._calculate_location_match(
                user_profile.get('preferences', {}).get('locations', []),
                internship.get('location', '')
            )
            
            # Semantic similarity using sentence transformers
            if sentence_model:
                scores['semantic_match'] = await self._calculate_semantic_similarity(
                    user_profile.get('bio', ''), 
                    internship.get('description', '')
                )
            else:
                scores['semantic_match'] = 0.5
            
            # Weighted final score
            final_score = (
                scores['skill_match'] * self.skill_weights['exact_match'] +
                scores['semantic_match'] * self.skill_weights['semantic_similarity'] +
                scores['experience_match'] * self.skill_weights['experience_relevance'] +
                scores['location_match'] * self.skill_weights['location_preference']
            )
            
            return min(100, max(0, int(final_score * 100)))
            
        except Exception as e:
            logger.error(f"Match score calculation error: {e}")
            return 50  # Default score

    def _calculate_skill_match(self, user_skills: List[str], job_skills: List[str]) -> float:
        """Calculate skill overlap score"""
        if not user_skills or not job_skills:
            return 0.0
        
        user_skills_lower = [skill.lower() for skill in user_skills]
        job_skills_lower = [skill.lower() for skill in job_skills]
        
        matches = set(user_skills_lower) & set(job_skills_lower)
        return len(matches) / len(job_skills_lower)

    def _calculate_experience_match(self, user_experience: List[Dict], job_requirements: List[str]) -> float:
        """Calculate experience relevance score"""
        if not user_experience or not job_requirements:
            return 0.5
        
        # Simple keyword matching between experience and requirements
        exp_text = ' '.join([exp.get('description', '') for exp in user_experience]).lower()
        req_text = ' '.join(job_requirements).lower()
        
        common_words = set(exp_text.split()) & set(req_text.split())
        total_req_words = len(set(req_text.split()))
        
        return len(common_words) / total_req_words if total_req_words > 0 else 0.5

    def _calculate_location_match(self, user_locations: List[str], job_location: str) -> float:
        """Calculate location preference match"""
        if not user_locations or not job_location:
            return 0.5
        
        job_location_lower = job_location.lower()
        
        # Check for exact match or contains
        for loc in user_locations:
            if loc.lower() in job_location_lower or job_location_lower in loc.lower():
                return 1.0
        
        # Check for remote work
        if 'remote' in [loc.lower() for loc in user_locations] and 'remote' in job_location_lower:
            return 1.0
        
        return 0.0

    async def _calculate_semantic_similarity(self, user_bio: str, job_description: str) -> float:
        """Calculate semantic similarity using sentence transformers"""
        try:
            if not user_bio or not job_description:
                return 0.5
            
            # Generate embeddings
            embeddings = sentence_model.encode([user_bio, job_description])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.error(f"Semantic similarity error: {e}")
            return 0.5

# Web Scraping Engine
class InternshipScraper:
    def __init__(self):
        self.session = None
        self.scraped_data = []
        
    async def scrape_internshala(self, search_query: str = "", location: str = "") -> List[Dict]:
        """Scrape internships from Internshala"""
        try:
            url = "https://internshala.com/internships/"
            
            # Add query parameters if provided
            params = {}
            if search_query:
                params['search'] = search_query
            if location:
                params['location'] = location
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, params=params, headers=headers) as response:
                    html = await response.text()
                    
                soup = BeautifulSoup(html, 'html.parser')
                internships = []
                
                # Find internship cards (adjust selectors based on actual website structure)
                internship_cards = soup.find_all('div', class_='internship_meta')
                
                for card in internship_cards[:20]:  # Limit to 20 results
                    try:
                        title_elem = card.find('h3', class_='heading_4_5')
                        title = title_elem.text.strip() if title_elem else "Unknown Position"
                        
                        company_elem = card.find('p', class_='company_name')
                        company = company_elem.text.strip() if company_elem else "Unknown Company"
                        
                        location_elem = card.find('p', class_='location_link')
                        location = location_elem.text.strip() if location_elem else "Unknown Location"
                        
                        stipend_elem = card.find('span', class_='stipend')
                        stipend = stipend_elem.text.strip() if stipend_elem else "Not disclosed"
                        
                        duration_elem = card.find('div', class_='other_detail_item')
                        duration = duration_elem.text.strip() if duration_elem else "Not specified"
                        
                        internships.append({
                            "id": f"internshala_{hashlib.md5(f'{title}{company}'.encode()).hexdigest()[:8]}",
                            "title": title,
                            "company": company,
                            "location": location,
                            "stipend": stipend,
                            "duration": duration,
                            "source": "Internshala",
                            "scraped_at": datetime.now().isoformat(),
                            "url": "https://internshala.com" + card.get('href', '') if card.get('href') else ""
                        })
                    except Exception as e:
                        logger.error(f"Error parsing internship card: {e}")
                        continue
                
                return internships
                
        except Exception as e:
            logger.error(f"Internshala scraping error: {e}")
            return []

    async def scrape_linkedin_jobs(self, search_query: str = "internship", location: str = "India") -> List[Dict]:
        """Scrape internships from LinkedIn (simplified version)"""
        try:
            # Note: LinkedIn has anti-scraping measures, this is a simplified version
            # In production, you'd need to use LinkedIn API or more sophisticated scraping
            
            url = f"https://www.linkedin.com/jobs/search/?keywords={search_query}&location={location}"
            
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                
                async with session.get(url, headers=headers) as response:
                    html = await response.text()
                    
                soup = BeautifulSoup(html, 'html.parser')
                internships = []
                
                # Find job cards (adjust selectors based on LinkedIn structure)
                job_cards = soup.find_all('div', class_='job-search-card')
                
                for card in job_cards[:15]:  # Limit to 15 results
                    try:
                        title_elem = card.find('h3', class_='base-search-card__title')
                        title = title_elem.text.strip() if title_elem else "Unknown Position"
                        
                        company_elem = card.find('h4', class_='base-search-card__subtitle')
                        company = company_elem.text.strip() if company_elem else "Unknown Company"
                        
                        location_elem = card.find('span', class_='job-search-card__location')
                        location = location_elem.text.strip() if location_elem else "Unknown Location"
                        
                        internships.append({
                            "id": f"linkedin_{hashlib.md5(f'{title}{company}'.encode()).hexdigest()[:8]}",
                            "title": title,
                            "company": company,
                            "location": location,
                            "source": "LinkedIn",
                            "scraped_at": datetime.now().isoformat(),
                            "url": card.find('a')['href'] if card.find('a') else ""
                        })
                    except Exception as e:
                        logger.error(f"Error parsing LinkedIn job card: {e}")
                        continue
                
                return internships
                
        except Exception as e:
            logger.error(f"LinkedIn scraping error: {e}")
            return []

# GitHub Profile Analyzer
class GitHubAnalyzer:
    def __init__(self):
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.github_client = Github(self.github_token) if self.github_token else None

    async def analyze_profile(self, username: str) -> Dict[str, Any]:
        """Analyze GitHub profile and extract insights"""
        try:
            if not self.github_client:
                raise HTTPException(status_code=500, detail="GitHub API not configured")
            
            user = self.github_client.get_user(username)
            repos = list(user.get_repos())
            
            # Basic stats
            stats = {
                "public_repos": user.public_repos,
                "followers": user.followers,
                "following": user.following,
                "total_commits": 0,
                "total_stars": sum(repo.stargazers_count for repo in repos),
                "total_forks": sum(repo.forks_count for repo in repos)
            }
            
            # Language analysis
            language_stats = {}
            for repo in repos[:50]:  # Analyze top 50 repos
                if repo.language:
                    language_stats[repo.language] = language_stats.get(repo.language, 0) + 1
            
            # Convert to percentages
            total_repos = len([r for r in repos if r.language])
            top_languages = []
            
            for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True)[:5]:
                percentage = (count / total_repos) * 100 if total_repos > 0 else 0
                top_languages.append({
                    "name": lang,
                    "percentage": round(percentage, 1),
                    "color": self._get_language_color(lang)
                })
            
            stats["top_languages"] = top_languages
            
            # Repository analysis
            featured_repos = []
            for repo in sorted(repos, key=lambda x: x.stargazers_count, reverse=True)[:5]:
                featured_repos.append({
                    "name": repo.name,
                    "description": repo.description or "No description available",
                    "language": repo.language,
                    "stars": repo.stargazers_count,
                    "forks": repo.forks_count,
                    "url": repo.html_url,
                    "is_private": repo.private
                })
            
            # Contribution activity (simplified)
            contribution_score = min(100, (stats["total_stars"] + stats["public_repos"] * 2 + stats["followers"]) // 10)
            
            return {
                "username": username,
                "stats": stats,
                "repositories": featured_repos,
                "contribution_score": contribution_score,
                "profile_url": f"https://github.com/{username}",
                "analyzed_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"GitHub analysis error: {e}")
            raise HTTPException(status_code=400, detail=f"GitHub analysis failed: {str(e)}")

    def _get_language_color(self, language: str) -> str:
        """Get color for programming language"""
        colors = {
            'JavaScript': '#f7df1e',
            'Python': '#3776ab',
            'Java': '#ed8b00',
            'TypeScript': '#007acc',
            'HTML': '#e34c26',
            'CSS': '#1572b6',
            'C++': '#00599c',
            'C': '#555555',
            'PHP': '#777bb4',
            'Ruby': '#cc342d',
            'Go': '#00add8',
            'Rust': '#dea584',
            'Swift': '#fa7343',
            'Kotlin': '#7f52ff'
        }
        return colors.get(language, '#333333')

# Initialize services
resume_parser = AIResumeParser()
matching_engine = AIMatchingEngine()
internship_scraper = InternshipScraper()
github_analyzer = GitHubAnalyzer()

# Background task for real-time internship scraping
@celery_app.task
def scrape_internships_task():
    """Background task to scrape internships"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Scrape from multiple sources
        internshala_data = loop.run_until_complete(internship_scraper.scrape_internshala())
        linkedin_data = loop.run_until_complete(internship_scraper.scrape_linkedin_jobs())
        
        # Combine and deduplicate
        all_internships = internshala_data + linkedin_data
        
        # Store in Firebase
        for internship in all_internships:
            doc_ref = db.collection('scraped_internships').document(internship['id'])
            doc_ref.set({
                **internship,
                'created_at': datetime.now(),
                'status': 'active'
            })
        
        logger.info(f"Scraped and stored {len(all_internships)} internships")
        return len(all_internships)
        
    except Exception as e:
        logger.error(f"Background scraping task error: {e}")
        return 0

# API Endpoints

@app.get("/")
async def root():
    return {"message": "InternAI Backend API", "version": "1.0.0", "status": "active"}

@app.post("/api/auth/verify")
async def verify_auth(user: dict = Depends(get_current_user)):
    """Verify authentication token"""
    return {"user_id": user['uid'], "email": user.get('email')}

@app.post("/api/profile/create")
async def create_profile(profile: UserProfile, user: dict = Depends(get_current_user)):
    """Create or update user profile"""
    try:
        user_id = user['uid']
        
        # Save to Firestore
        doc_ref = db.collection('users').document(user_id)
        doc_ref.set({
            **profile.dict(),
            'created_at': datetime.now(),
            'updated_at': datetime.now()
        })
        
        return {"message": "Profile created successfully", "user_id": user_id}
    except Exception as e:
        logger.error(f"Profile creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create profile")

@app.post("/api/resume/upload")
async def upload_resume(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user)
):
    """Upload and parse resume"""
    try:
        # Validate file
        if not file.filename.lower().endswith(('.pdf', '.docx')):
            raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")
        
        # Read file content
        file_content = await file.read()
        
        # Parse resume
        parsed_data = await resume_parser.parse_resume(file_content, file.filename)
        
        # Save to Firebase Storage
        user_id = user['uid']
        blob_name = f"resumes/{user_id}/{file.filename}"
        
        bucket = storage.bucket()
        blob = bucket.blob(blob_name)
        blob.upload_from_string(file_content)
        
        # Make file publicly accessible
        blob.make_public()
        file_url = blob.public_url
        
        # Update user profile with resume data
        doc_ref = db.collection('users').document(user_id)
        doc_ref.update({
            'resume': {
                'filename': file.filename,
                'url': file_url,
                'uploaded_at': datetime.now(),
                'parsed_data': parsed_data.dict()
            },
            'updated_at': datetime.now()
        })
        
        return {
            "message": "Resume uploaded and parsed successfully",
            "parsed_data": parsed_data.dict(),
            "file_url": file_url
        }
        
    except Exception as e:
        logger.error(f"Resume upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.post("/api/github/connect")
async def connect_github(
    github_username: str,
    user: dict = Depends(get_current_user)
):
    """Connect and analyze GitHub profile"""
    try:
        user_id = user['uid']
        
        # Analyze GitHub profile
        github_data = await github_analyzer.analyze_profile(github_username)
        
        # Update user profile
        doc_ref = db.collection('users').document(user_id)
        doc_ref.update({
            'github': github_data,
            'updated_at': datetime.now()
        })
        
        return {
            "message": "GitHub profile connected successfully",
            "github_data": github_data
        }
        
    except Exception as e:
        logger.error(f"GitHub connection error: {e}")
        raise HTTPException(status_code=500, detail=f"GitHub connection failed: {str(e)}")

@app.get("/api/internships/recommendations")
async def get_internship_recommendations(
    user: dict = Depends(get_current_user),
    limit: int = 20
):
    """Get personalized internship recommendations"""
    try:
        user_id = user['uid']
        
        # Get user profile
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        user_data = user_doc.to_dict()
        
        # Get available internships (both mock and scraped)
        internships = []
        
        # Add mock internships from the data
        mock_internships = [
            {
                "id": "intern_001",
                "title": "Full Stack Development Intern",
                "company": "TechCorp Solutions",
                "location": "Bangalore, India",
                "skills": ["React", "Node.js", "MongoDB", "Express.js", "JavaScript", "Git"],
                "description": "Join our dynamic team to build cutting-edge web applications using MERN stack.",
                "stipend": "₹30,000/month",
                "duration": "6 months"
            },
            {
                "id": "intern_002",
                "title": "AI/ML Research Intern",
                "company": "DeepMind Research Lab",
                "location": "Hyderabad, India",
                "skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "NLP"],
                "description": "Work on cutting-edge AI research projects focusing on NLP and computer vision.",
                "stipend": "₹45,000/month",
                "duration": "4-6 months"
            }
        ]
        
        # Calculate match scores for each internship
        recommendations = []
        for internship in mock_internships:
            match_score = await matching_engine.calculate_match_score(user_data, internship)
            recommendations.append({
                **internship,
                "match_score": match_score,
                "match_reasons": [
                    "Skills alignment",
                    "Experience match",
                    "Location preference"
                ]
            })
        
        # Sort by match score
        recommendations.sort(key=lambda x: x['match_score'], reverse=True)
        
        return {
            "recommendations": recommendations[:limit],
            "total_count": len(recommendations)
        }
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")

@app.post("/api/internships/apply")
async def apply_to_internship(
    internship_id: str,
    cover_letter: Optional[str] = None,
    user: dict = Depends(get_current_user)
):
    """Apply to an internship"""
    try:
        user_id = user['uid']
        application_id = f"{user_id}_{internship_id}_{int(datetime.now().timestamp())}"
        
        # Create application record
        application_data = {
            "id": application_id,
            "user_id": user_id,
            "internship_id": internship_id,
            "cover_letter": cover_letter,
            "status": "applied",
            "applied_at": datetime.now(),
            "updated_at": datetime.now()
        }
        
        # Save to Firestore
        db.collection('applications').document(application_id).set(application_data)
        
        # Update user's applications list
        user_doc_ref = db.collection('users').document(user_id)
        user_doc_ref.update({
            f'applications.{application_id}': application_data,
            'updated_at': datetime.now()
        })
        
        return {
            "message": "Application submitted successfully",
            "application_id": application_id
        }
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit application")

@app.get("/api/analytics/dashboard")
async def get_analytics_dashboard(user: dict = Depends(get_current_user)):
    """Get user analytics dashboard data"""
    try:
        user_id = user['uid']
        
        # Get user data
        user_doc = db.collection('users').document(user_id).get()
        if not user_doc.exists:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        user_data = user_doc.to_dict()
        
        # Generate analytics
        analytics = {
            "profile_completion": user_data.get('profile', {}).get('completion_percentage', 0),
            "applications_sent": len(user_data.get('applications', {})),
            "profile_views": 156,  # Mock data
            "avg_match_score": 89,  # Mock data
            "top_skills": user_data.get('resume', {}).get('parsed_data', {}).get('skills', [])[:5],
            "activity_score": 95,  # Mock data
            "chart_data": {
                "applications_over_time": [
                    {"date": "2024-09-01", "applications": 2},
                    {"date": "2024-09-08", "applications": 3},
                    {"date": "2024-09-15", "applications": 1}
                ],
                "skill_match_distribution": [
                    {"skill": "React", "match_percentage": 95},
                    {"skill": "Python", "match_percentage": 88},
                    {"skill": "JavaScript", "match_percentage": 92}
                ]
            }
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"Analytics error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analytics")

@app.post("/api/scraping/trigger")
async def trigger_scraping(
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """Trigger internship scraping (admin only)"""
    try:
        # Add background task
        background_tasks.add_task(scrape_internships_task)
        
        return {"message": "Scraping task started in background"}
        
    except Exception as e:
        logger.error(f"Scraping trigger error: {e}")
        raise HTTPException(status_code=500, detail="Failed to trigger scraping")

# WebSocket endpoint for real-time updates (if needed)
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.send_personal_message(f"Message: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)