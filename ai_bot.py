"""
AI Bot Integration File
This file shows how to integrate your AI bot with the backend
Replace the mock functions with your actual AI model calls
"""

import sys
import json
import random
from typing import Dict, List, Any

def analyze_resume_with_ai(resume_text: str, resume_data: Dict) -> Dict:
    """
    Replace this with your actual AI model for resume analysis
    This is a mock implementation showing the expected structure
    """
    # Mock AI analysis - replace with your model
    skills = resume_data.get("skills", [])
    
    # AI-enhanced skill extraction
    enhanced_skills = skills + ["Problem Solving", "Communication", "Teamwork"]
    
    # AI role prediction
    roles = []
    skills_text = " ".join(skills).lower()
    
    if any(skill in skills_text for skill in ["python", "machine learning", "data"]):
        roles.append("Data Science Intern")
    if any(skill in skills_text for skill in ["html", "css", "javascript", "react"]):
        roles.append("Frontend Developer Intern")
    if any(skill in skills_text for skill in ["node.js", "django", "flask", "sql"]):
        roles.append("Backend Developer Intern")
    
    if not roles:
        roles = ["Software Developer Intern"]
    
    # AI compatibility scoring
    compatibility_score = min(95, 70 + len(skills) * 3)
    
    return {
        "enhanced_skills": enhanced_skills,
        "predicted_roles": roles,
        "compatibility_score": compatibility_score,
        "experience_level": determine_experience_level(resume_data),
        "recommendations": generate_ai_recommendations(resume_data),
        "career_insights": generate_career_insights(skills, roles)
    }

def generate_ai_chat_response(message: str, context: Dict) -> str:
    """
    Replace this with your actual AI chatbot model
    This is a mock implementation
    """
    message_lower = message.lower()
    user_skills = context.get("user_skills", [])
    resume_uploaded = context.get("resume_uploaded", False)
    
    # Mock AI responses - replace with your model
    if "resume" in message_lower:
        if resume_uploaded:
            return f"Great! I've analyzed your resume with {len(user_skills)} skills. Your profile shows strong potential for roles in software development. Would you like me to find matching internships?"
        else:
            return "I'd be happy to analyze your resume! Upload it using the orange button below the header, and I'll extract your skills, predict suitable roles, and find perfect internship matches."
    
    elif "internship" in message_lower or "job" in message_lower:
        if resume_uploaded:
            return "Based on your skills, I've found high-quality internships with competitive stipends up to ₹25,000. The matches are personalized based on your experience level and skill set. Check the Discover section!"
        else:
            return "Upload your resume first, and I'll find personalized internships that match your skills and experience level perfectly!"
    
    elif "salary" in message_lower or "stipend" in message_lower:
        return "Internship stipends vary by company tier and your experience:\n• Tier 1 (Google, Microsoft): ₹15,000-₹25,000\n• Tier 2 (Flipkart, Zomato): ₹10,000-₹20,000\n• Tier 3 (TCS, Infosys): ₹6,000-₹15,000\n\nYour exact offer depends on skill match and experience level!"
    
    elif "help" in message_lower:
        return """I'm your AI career assistant! I can help with:

🎯 **Resume Analysis** - Extract skills, predict roles, calculate compatibility
📊 **Smart Matching** - Find internships with 75-95% compatibility scores  
💰 **Salary Insights** - Realistic stipend ranges up to ₹25,000
🚀 **Career Guidance** - Personalized recommendations based on your profile
📈 **Application Strategy** - Tips to improve your success rate

What would you like to explore first?"""
    
    else:
        responses = [
            "I'm here to help you find the perfect internship! What specific aspect would you like assistance with?",
            f"Hello! As your AI career assistant, I can analyze resumes, find matching internships, and provide salary insights. How can I help?",
            "I specialize in connecting talented individuals like you with great internship opportunities. What can I assist you with today?"
        ]
        return random.choice(responses)

def determine_experience_level(resume_data: Dict) -> str:
    """AI-powered experience level determination"""
    skills_count = len(resume_data.get("skills", []))
    projects_text = resume_data.get("projects", "").lower()
    education_text = resume_data.get("education", "").lower()
    
    experience_score = 0
    
    # Skill diversity
    if skills_count >= 8:
        experience_score += 2
    elif skills_count >= 5:
        experience_score += 1
    
    # Project complexity indicators
    if "internship" in projects_text or "work" in projects_text:
        experience_score += 2
    if projects_text.count("project") >= 3:
        experience_score += 1
    
    # Education level
    if "master" in education_text or "graduate" in education_text:
        experience_score += 2
    elif "bachelor" in education_text:
        experience_score += 1
    
    # Experience level mapping
    if experience_score >= 5:
        return "experienced"
    elif experience_score >= 3:
        return "intermediate" 
    else:
        return "entry"

def generate_ai_recommendations(resume_data: Dict) -> List[str]:
    """Generate AI-powered improvement recommendations"""
    recommendations = []
    
    skills = resume_data.get("skills", [])
    projects = resume_data.get("projects", "")
    
    if len(skills) < 5:
        recommendations.append("Add more technical skills to increase your marketability")
    
    if not projects or len(projects.split()) < 50:
        recommendations.append("Include detailed project descriptions with technologies used")
    
    if not resume_data.get("linkedin"):
        recommendations.append("Add your LinkedIn profile URL for better networking")
    
    recommendations.extend([
        "Quantify your achievements with specific numbers and metrics",
        "Ensure your resume is ATS-compatible for better screening",
        "Consider adding relevant certifications to boost your profile"
    ])
    
    return recommendations[:4]  # Return top 4 recommendations

def generate_career_insights(skills: List[str], roles: List[str]) -> Dict:
    """Generate AI-powered career insights"""
    return {
        "market_demand": "High" if len(skills) >= 6 else "Medium",
        "skill_gaps": ["Communication", "Leadership", "Project Management"],
        "growth_potential": "Excellent" if "Data Science" in str(roles) else "Good",
        "recommended_next_steps": [
            "Build a strong GitHub portfolio",
            "Contribute to open-source projects", 
            "Network with industry professionals",
            "Consider relevant online certifications"
        ]
    }

def calculate_internship_match_score(user_skills: List[str], job_requirements: List[str], experience_level: str) -> int:
    """AI-powered match score calculation"""
    # Skill overlap calculation
    user_skills_lower = [skill.lower() for skill in user_skills]
    requirements_lower = [req.lower() for req in job_requirements]
    
    skill_overlap = len(set(user_skills_lower) & set(requirements_lower))
    total_requirements = len(job_requirements)
    
    # Base score from skill match
    base_score = min(90, (skill_overlap / total_requirements) * 100) if total_requirements > 0 else 50
    
    # Experience level adjustment
    if experience_level == "experienced":
        base_score = min(95, base_score + 5)
    elif experience_level == "entry":
        base_score = max(75, base_score - 5)
    
    # Add some intelligent randomness
    final_score = int(min(95, max(75, base_score + random.randint(-2, 2))))
    
    return final_score

# Command line interface for integration testing
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ai_bot.py <function_name> [args...]")
        sys.exit(1)
    
    function_name = sys.argv[1]
    
    if function_name == "analyze_resume":
        # Mock resume data for testing
        mock_resume_data = {
            "skills": ["Python", "JavaScript", "React", "SQL"],
            "projects": "Built a web application using React and Node.js. Created a machine learning model for data analysis.",
            "education": "Bachelor of Technology in Computer Science",
            "linkedin": "https://linkedin.com/in/testuser"
        }
        
        result = analyze_resume_with_ai("mock resume text", mock_resume_data)
        print(json.dumps(result, indent=2))
    
    elif function_name == "chat_response":
        message = sys.argv[2] if len(sys.argv) > 2 else "Help me find internships"
        context = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        
        response = generate_ai_chat_response(message, context)
        print(json.dumps({"response": response}))
    
    elif function_name == "match_score":
        user_skills = ["Python", "React", "JavaScript"]
        job_requirements = ["Python", "Machine Learning", "SQL"]
        experience_level = "intermediate"
        
        score = calculate_internship_match_score(user_skills, job_requirements, experience_level)
        print(json.dumps({"match_score": score}))
    
    else:
        print(f"Unknown function: {function_name}")
        sys.exit(1)