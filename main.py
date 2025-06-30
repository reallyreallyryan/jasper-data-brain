"""
Healthcare SEO Intelligence Engine
Specialized for SEMrush Organic Research exports
Analyzes keywords to find ranking opportunities and traffic growth potential
"""

import pandas as pd
import numpy as np
import json
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel
import uvicorn
import io


# Specialized Data Models for Ranking Intelligence
class RankingKeyword(BaseModel):
    keyword: str
    current_position: int
    previous_position: Optional[int]
    position_change: Optional[int]
    search_volume: int
    keyword_difficulty: int
    cpc: float
    url: str
    traffic: int
    traffic_percent: float
    traffic_cost: float
    serp_features: Optional[str]
    intent: Optional[str]
    opportunity_score: Optional[float]

class RankingOpportunity(BaseModel):
    category: str
    keywords: List[RankingKeyword]
    potential_traffic_gain: int
    potential_revenue_gain: float
    avg_difficulty: float
    recommendation: str

class RankingIntelligence(BaseModel):
    site_domain: str
    total_keywords: int
    total_traffic: int
    total_traffic_value: float
    avg_position: float
    
    # Opportunity Analysis
    quick_wins: List[RankingKeyword]           # Position 4-10, can reach top 3
    content_gaps: List[RankingKeyword]         # Position 11-20, need content work
    long_tail_opportunities: List[RankingKeyword]  # Position 21-50, long-tail focus
    declining_keywords: List[RankingKeyword]   # Lost positions, need attention
    
    # Revenue Analysis
    high_value_targets: List[RankingKeyword]   # High CPC + volume, revenue potential
    traffic_winners: List[RankingKeyword]      # Gaining positions + traffic
    
    # Content Strategy
    serp_feature_opportunities: List[RankingKeyword]  # Can target featured snippets
    intent_breakdown: Dict[str, int]
    
    # Strategic Insights
    insights: List[str]
    action_items: List[str]
    monthly_opportunity: Dict[str, Any]


class HealthcareRankingIntelligence:
    """Specialized ranking intelligence for healthcare/medical websites"""
    
    def __init__(self):
        self.healthcare_terms = [
            'pain', 'treatment', 'therapy', 'doctor', 'clinic', 'medical', 
            'spine', 'back', 'neck', 'surgery', 'orthopedic', 'sports',
            'injury', 'rehabilitation', 'physical therapy', 'chiropractor'
        ]
    
    def _extract_domain_from_url(self, df: pd.DataFrame) -> str:
        """Extract the main domain from URL column"""
        try:
            first_url = df['URL'].iloc[0]
            from urllib.parse import urlparse
            domain = urlparse(first_url).netloc
            return domain.replace('www.', '')
        except:
            return "your-website.com"
    
    def _calculate_opportunity_score(self, row) -> float:
        """Calculate opportunity score based on position, volume, difficulty, and CPC"""
        try:
            position = row['Position']
            volume = row['Search Volume']
            difficulty = row['Keyword Difficulty']
            cpc = row['CPC']
            
            # Base score calculation
            # Higher volume = better
            volume_score = min(volume / 1000, 10)  # Cap at 10
            
            # Lower position = higher opportunity (if not in top 3)
            if position <= 3:
                position_score = 2  # Already doing well
            elif position <= 10:
                position_score = 8  # Quick win opportunity
            elif position <= 20:
                position_score = 6  # Content improvement opportunity
            else:
                position_score = 3  # Long-term opportunity
            
            # Lower difficulty = easier to improve
            difficulty_score = max(0, 10 - (difficulty / 10))
            
            # Higher CPC = more valuable
            cpc_score = min(cpc * 2, 10)  # Cap at 10
            
            # Weighted average
            opportunity_score = (
                volume_score * 0.3 +
                position_score * 0.4 +
                difficulty_score * 0.2 +
                cpc_score * 0.1
            )
            
            return round(opportunity_score, 2)
            
        except:
            return 0.0
    
    def _categorize_keywords(self, df: pd.DataFrame) -> Dict[str, List[RankingKeyword]]:
        """Categorize keywords into strategic buckets"""
        
        # Add opportunity scores
        df['Opportunity_Score'] = df.apply(self._calculate_opportunity_score, axis=1)
        
        categories = {
            'quick_wins': [],
            'content_gaps': [],
            'long_tail_opportunities': [],
            'declining_keywords': [],
            'high_value_targets': [],
            'traffic_winners': [],
            'serp_feature_opportunities': []
        }
        
        for _, row in df.iterrows():
            keyword_data = RankingKeyword(
                keyword=str(row['Keyword']),
                current_position=int(row['Position']),
                previous_position=int(row['Previous position']) if pd.notna(row['Previous position']) else None,
                position_change=(int(row['Previous position']) - int(row['Position'])) if pd.notna(row['Previous position']) else None,
                search_volume=int(row['Search Volume']),
                keyword_difficulty=int(row['Keyword Difficulty']),
                cpc=float(row['CPC']),
                url=str(row['URL']),
                traffic=int(row['Traffic']),
                traffic_percent=float(row['Traffic (%)']),
                traffic_cost=float(row['Traffic Cost']),
                serp_features=str(row['SERP Features by Keyword']) if pd.notna(row['SERP Features by Keyword']) else None,
                intent=str(row['Keyword Intents']) if pd.notna(row['Keyword Intents']) else None,
                opportunity_score=row['Opportunity_Score']
            )
            
            position = int(row['Position'])
            prev_position = int(row['Previous position']) if pd.notna(row['Previous position']) else None
            volume = int(row['Search Volume'])
            cpc = float(row['CPC'])
            
            # Quick Wins: Position 4-10, good volume
            if 4 <= position <= 10 and volume >= 100:
                categories['quick_wins'].append(keyword_data)
            
            # Content Gaps: Position 11-20
            if 11 <= position <= 20 and volume >= 50:
                categories['content_gaps'].append(keyword_data)
            
            # Long-tail Opportunities: Position 21-50
            if 21 <= position <= 50:
                categories['long_tail_opportunities'].append(keyword_data)
            
            # Declining Keywords: Lost 3+ positions
            if prev_position and (position - prev_position) >= 3:
                categories['declining_keywords'].append(keyword_data)
            
            # High Value Targets: High CPC + decent volume
            if cpc >= 2.0 and volume >= 100:
                categories['high_value_targets'].append(keyword_data)
            
            # Traffic Winners: Gained positions
            if prev_position and (prev_position - position) >= 1:
                categories['traffic_winners'].append(keyword_data)
            
            # SERP Feature Opportunities
            serp_features = str(row['SERP Features by Keyword']) if pd.notna(row['SERP Features by Keyword']) else ""
            if any(feature in serp_features for feature in ['Featured Snippet', 'People Also Ask']) and position <= 10:
                categories['serp_feature_opportunities'].append(keyword_data)
        
        # Sort each category by opportunity score
        for category in categories:
            categories[category] = sorted(categories[category], 
                                       key=lambda x: x.opportunity_score, 
                                       reverse=True)[:20]  # Top 20 per category
        
        return categories
    
    def _generate_healthcare_insights(self, df: pd.DataFrame, categories: Dict) -> List[str]:
        """Generate healthcare-specific SEO insights"""
        insights = []
        
        total_keywords = len(df)
        avg_position = df['Position'].mean()
        total_traffic = df['Traffic'].sum()
        total_value = df['Traffic Cost'].sum()
        
        insights.append(f"📊 Tracking {total_keywords:,} healthcare keywords with average position {avg_position:.1f}")
        insights.append(f"🚀 Current monthly traffic: {total_traffic:,} visitors worth ${total_value:,.2f}")
        
        # Quick wins analysis
        quick_wins = len(categories['quick_wins'])
        if quick_wins > 0:
            avg_qw_volume = np.mean([kw.search_volume for kw in categories['quick_wins']])
            insights.append(f"🎯 {quick_wins} quick-win opportunities (avg {avg_qw_volume:.0f} monthly searches)")
        
        # Content gaps
        content_gaps = len(categories['content_gaps'])
        if content_gaps > 0:
            insights.append(f"📝 {content_gaps} content improvement opportunities in positions 11-20")
        
        # Declining keywords
        declining = len(categories['declining_keywords'])
        if declining > 0:
            insights.append(f"⚠️ {declining} keywords declining - need immediate attention")
        
        # High-value analysis
        high_value = categories['high_value_targets']
        if high_value:
            potential_value = sum(kw.cpc * kw.search_volume * 0.02 for kw in high_value[:10])  # 2% CTR estimate
            insights.append(f"💰 Top 10 high-value targets could generate ${potential_value:,.2f}/month")
        
        # Healthcare-specific insights
        healthcare_kws = df[df['Keyword'].str.contains('|'.join(self.healthcare_terms), case=False, na=False)]
        if len(healthcare_kws) > 0:
            healthcare_traffic = healthcare_kws['Traffic'].sum()
            healthcare_percent = (healthcare_traffic / total_traffic) * 100
            insights.append(f"🏥 {len(healthcare_kws)} healthcare keywords driving {healthcare_percent:.1f}% of traffic")
        
        # SERP features
        serp_opps = len(categories['serp_feature_opportunities'])
        if serp_opps > 0:
            insights.append(f"🎯 {serp_opps} featured snippet opportunities in top 10 positions")
        
        return insights
    
    def _generate_action_items(self, categories: Dict) -> List[str]:
        """Generate specific action items"""
        actions = []
        
        # Quick wins
        if categories['quick_wins']:
            top_qw = categories['quick_wins'][0]
            actions.append(f"🚀 PRIORITY: Optimize '{top_qw.keyword}' (pos {top_qw.current_position}) - could reach top 3")
        
        # Content gaps
        if categories['content_gaps']:
            actions.append(f"📝 Create comprehensive content for {len(categories['content_gaps'])} keywords in positions 11-20")
        
        # Declining keywords
        if categories['declining_keywords']:
            worst_decline = max(categories['declining_keywords'], key=lambda x: x.position_change or 0)
            actions.append(f"⚠️ URGENT: Fix '{worst_decline.keyword}' - dropped {worst_decline.position_change} positions")
        
        # High-value targets
        if categories['high_value_targets']:
            top_value = categories['high_value_targets'][0]
            actions.append(f"💰 Focus on '{top_value.keyword}' (${top_value.cpc:.2f} CPC, {top_value.search_volume} volume)")
        
        # SERP features
        if categories['serp_feature_opportunities']:
            actions.append(f"🎯 Optimize for featured snippets: {len(categories['serp_feature_opportunities'])} opportunities")
        
        return actions
    
    async def analyze_ranking_data(self, file_path: str) -> RankingIntelligence:
        """Main analysis function for SEMrush Organic Research exports"""
        try:
            # Read the CSV
            df = pd.read_csv(file_path)
            
            print(f"📊 Loaded ranking data: {len(df)} keywords")
            print(f"📋 Columns: {list(df.columns)}")
            
            # Clean and prepare data
            # Clean and prepare data more aggressively
            df = df.dropna(subset=['Keyword', 'Position', 'Search Volume'])

            # Replace any non-numeric values with defaults
            df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            df['Search Volume'] = pd.to_numeric(df['Search Volume'], errors='coerce')
            df['Keyword Difficulty'] = pd.to_numeric(df['Keyword Difficulty'], errors='coerce').fillna(0)
            df['CPC'] = pd.to_numeric(df['CPC'], errors='coerce').fillna(0)
            df['Traffic'] = pd.to_numeric(df['Traffic'], errors='coerce').fillna(0)
            df['Traffic (%)'] = pd.to_numeric(df['Traffic (%)'], errors='coerce').fillna(0)
            df['Traffic Cost'] = pd.to_numeric(df['Traffic Cost'], errors='coerce').fillna(0)

            # Remove any rows where Position or Search Volume couldn't be converted
            df = df.dropna(subset=['Position', 'Search Volume'])

            # Convert to integers after cleaning
            df['Position'] = df['Position'].astype(int)
            df['Search Volume'] = df['Search Volume'].astype(int)
            df['Traffic'] = df['Traffic'].astype(int)
            
            # Extract domain
            domain = self._extract_domain_from_url(df)
            print(f"🌐 Analyzing rankings for: {domain}")
            
            # Calculate key metrics
            total_keywords = len(df)
            total_traffic = df['Traffic'].sum()
            total_traffic_value = df['Traffic Cost'].sum()
            avg_position = df['Position'].mean()
            
            # Categorize keywords
            print(f"🔍 Categorizing {total_keywords} keywords...")
            categories = self._categorize_keywords(df)
            
            # Intent breakdown
            intent_counts = df['Keyword Intents'].value_counts().to_dict() if 'Keyword Intents' in df.columns else {}
            
            # Generate insights
            print(f"🧠 Generating healthcare SEO insights...")
            insights = self._generate_healthcare_insights(df, categories)
            action_items = self._generate_action_items(categories)
            
            # Calculate monthly opportunity
            quick_win_traffic = sum(kw.search_volume * 0.05 for kw in categories['quick_wins'][:10])  # 5% CTR for top 3
            content_gap_traffic = sum(kw.search_volume * 0.02 for kw in categories['content_gaps'][:10])  # 2% CTR
            monthly_opportunity = {
                "potential_traffic_gain": int(quick_win_traffic + content_gap_traffic),
                "potential_revenue_gain": sum(kw.cpc * kw.search_volume * 0.03 for kw in categories['high_value_targets'][:5]),
                "quick_wins_count": len(categories['quick_wins']),
                "content_gaps_count": len(categories['content_gaps'])
            }
            
            print(f"✅ Analysis complete!")
            
            return RankingIntelligence(
                site_domain=domain,
                total_keywords=total_keywords,
                total_traffic=int(total_traffic),
                total_traffic_value=float(total_traffic_value),
                avg_position=float(avg_position),
                
                quick_wins=categories['quick_wins'],
                content_gaps=categories['content_gaps'],
                long_tail_opportunities=categories['long_tail_opportunities'],
                declining_keywords=categories['declining_keywords'],
                high_value_targets=categories['high_value_targets'],
                traffic_winners=categories['traffic_winners'],
                serp_feature_opportunities=categories['serp_feature_opportunities'],
                
                intent_breakdown=intent_counts,
                insights=insights,
                action_items=action_items,
                monthly_opportunity=monthly_opportunity
            )
            
        except Exception as e:
            print(f"🚨 Analysis error: {e}")
            raise HTTPException(status_code=500, detail=f"Ranking analysis failed: {str(e)}")


# FastAPI App
app = FastAPI(
    title="Healthcare SEO Intelligence Engine",
    description="Specialized SEO ranking analysis for healthcare websites using SEMrush Organic Research exports",
    version="2.0.0"
)

intelligence_engine = HealthcareRankingIntelligence()


@app.get("/")
async def root():
    return {
        "message": "🏥 Healthcare SEO Intelligence Engine",
        "status": "ready",
        "optimized_for": "SEMrush Organic Research Exports",
        "capabilities": [
            "Ranking Opportunity Analysis",
            "Quick Win Identification", 
            "Content Gap Analysis",
            "Revenue Impact Assessment",
            "Healthcare SEO Insights",
            "SERP Feature Optimization",
            "Position Change Tracking"
        ]
    }


@app.get("/upload-form", response_class=HTMLResponse)
async def upload_form():
    """Enhanced upload form for Healthcare SEO Ranking Intelligence"""
    return """
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🏥 Healthcare SEO - Ranking Intelligence Engine</title>
            <style>
                :root {
                    --primary-blue: #2563eb;
                    --primary-blue-dark: #1d4ed8;
                    --healthcare-red: #dc2626;
                    --success-green: #16a34a;
                    --light-gray: #f8fafc;
                    --border-gray: #e2e8f0;
                    --text-gray: #475569;
                }
                
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: var(--text-gray);
                    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    min-height: 100vh;
                }
                
                .container {
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 2rem;
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 3rem;
                    background: white;
                    padding: 2rem;
                    border-radius: 16px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }
                
                .header h1 {
                    color: var(--primary-blue);
                    font-size: 2.5rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                }
                
                .header .subtitle {
                    color: var(--healthcare-red);
                    font-size: 1.2rem;
                    font-weight: 600;
                    margin-bottom: 1rem;
                }
                
                .header .description {
                    color: var(--text-gray);
                    font-size: 1rem;
                    max-width: 600px;
                    margin: 0 auto;
                }
                
                .upload-section {
                    background: white;
                    border-radius: 16px;
                    padding: 3rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    border: 2px dashed var(--border-gray);
                    transition: all 0.3s ease;
                }
                
                .upload-section:hover {
                    border-color: var(--primary-blue);
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px -5px rgba(0, 0, 0, 0.1);
                }
                
                .upload-section h2 {
                    color: var(--primary-blue);
                    font-size: 1.8rem;
                    margin-bottom: 1rem;
                    text-align: center;
                }
                
                .file-input-container {
                    position: relative;
                    margin: 2rem 0;
                }
                
                .file-input {
                    width: 100%;
                    padding: 1rem;
                    border: 2px solid var(--border-gray);
                    border-radius: 8px;
                    font-size: 1rem;
                    transition: border-color 0.3s ease;
                }
                
                .file-input:focus {
                    outline: none;
                    border-color: var(--primary-blue);
                }
                
                .upload-instructions {
                    background: var(--light-gray);
                    padding: 1.5rem;
                    border-radius: 8px;
                    margin: 1.5rem 0;
                    border-left: 4px solid var(--primary-blue);
                }
                
                .upload-instructions strong {
                    color: var(--primary-blue);
                }
                
                .analyze-btn {
                    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
                    color: white;
                    padding: 1rem 2rem;
                    border: none;
                    border-radius: 8px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.3s ease;
                    display: block;
                    margin: 2rem auto 0;
                    min-width: 200px;
                }
                
                .analyze-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px -5px rgba(37, 99, 235, 0.4);
                }
                
                .analyze-btn:active {
                    transform: translateY(0);
                }
                
                .features-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 2rem;
                }
                
                .feature-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    transition: transform 0.3s ease;
                }
                
                .feature-card:hover {
                    transform: translateY(-4px);
                }
                
                .feature-card .icon {
                    font-size: 2rem;
                    margin-bottom: 1rem;
                }
                
                .feature-card h3 {
                    color: var(--primary-blue);
                    font-size: 1.2rem;
                    margin-bottom: 0.5rem;
                }
                
                .feature-card p {
                    color: var(--text-gray);
                    font-size: 0.9rem;
                }
                
                .capabilities {
                    background: white;
                    padding: 2rem;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                }
                
                .capabilities h3 {
                    color: var(--healthcare-red);
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                    text-align: center;
                }
                
                .capabilities-list {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1rem;
                    list-style: none;
                }
                
                .capabilities-list li {
                    padding: 0.75rem;
                    background: var(--light-gray);
                    border-radius: 6px;
                    border-left: 3px solid var(--success-green);
                }
                
                .capabilities-list strong {
                    color: var(--primary-blue);
                }
                
                .loading {
                    display: none;
                    text-align: center;
                    margin-top: 1rem;
                }
                
                .loading .spinner {
                    border: 3px solid #f3f3f3;
                    border-top: 3px solid var(--primary-blue);
                    border-radius: 50%;
                    width: 30px;
                    height: 30px;
                    animation: spin 1s linear infinite;
                    margin: 0 auto 1rem;
                }
                
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                
                @media (max-width: 768px) {
                    .container {
                        padding: 1rem;
                    }
                    
                    .header h1 {
                        font-size: 2rem;
                    }
                    
                    .upload-section {
                        padding: 2rem;
                    }
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Healthcare SEO Intelligence</h1>
                    <p class="subtitle">Advanced Ranking Analysis for Medical Practices</p>
                    <p class="description">
                        Transform your SEMrush data into actionable SEO strategies. 
                        Identify quick wins, content gaps, and revenue opportunities specifically for healthcare websites.
                    </p>
                </div>
                
                <form id="analysisForm" action="/analyze-rankings/" method="post" enctype="multipart/form-data">
                    <div class="upload-section">
                        <h2>📊 Upload Your SEMrush Export</h2>
                        
                        <div class="file-input-container">
                            <input type="file" name="file" accept=".csv" required class="file-input" id="csvFile">
                        </div>
                        
                        <div class="upload-instructions">
                            <p><strong>Expected Format:</strong> SEMrush → Organic Research → Export Organic Keywords</p>
                            <p>💡 <strong>Tip:</strong> Include both current and previous position data for trend analysis</p>
                        </div>
                        
                        <button type="submit" class="analyze-btn" id="analyzeBtn">
                            🚀 Analyze Rankings
                        </button>
                        
                        <div class="loading" id="loadingIndicator">
                            <div class="spinner"></div>
                            <p>Analyzing your rankings... This may take a few moments.</p>
                        </div>
                    </div>
                </form>
                
                <div class="features-grid">
                    <div class="feature-card">
                        <div class="icon">🎯</div>
                        <h3>Quick Wins</h3>
                        <p>Keywords in positions 4-10 that can reach top 3 with focused optimization</p>
                    </div>
                    <div class="feature-card">
                        <div class="icon">📝</div>
                        <h3>Content Gaps</h3>
                        <p>Positions 11-20 needing comprehensive content development</p>
                    </div>
                    <div class="feature-card">
                        <div class="icon">💰</div>
                        <h3>Revenue Opportunities</h3>
                        <p>High CPC keywords with significant traffic and revenue potential</p>
                    </div>
                    <div class="feature-card">
                        <div class="icon">⚠️</div>
                        <h3>Declining Keywords</h3>
                        <p>Keywords losing positions that require immediate attention</p>
                    </div>
                </div>
                
                <div class="capabilities">
                    <h3>🏥 Healthcare SEO Specializations</h3>
                    <ul class="capabilities-list">
                        <li><strong>Medical Keyword Analysis</strong> - Specialized for healthcare terminology and patient search behavior</li>
                        <li><strong>Patient Intent Classification</strong> - Distinguish between informational and appointment-seeking queries</li>
                        <li><strong>Local SEO Opportunities</strong> - "Near me" searches and location-based optimization</li>
                        <li><strong>Treatment Page Optimization</strong> - Service-specific ranking opportunities and content gaps</li>
                        <li><strong>Competition Analysis</strong> - Compare performance against other medical practices</li>
                        <li><strong>SERP Feature Targeting</strong> - Featured snippets and "People Also Ask" opportunities</li>
                    </ul>
                </div>
            </div>
            
            <script>
                document.getElementById('analysisForm').addEventListener('submit', function(e) {
                    const analyzeBtn = document.getElementById('analyzeBtn');
                    const loadingIndicator = document.getElementById('loadingIndicator');
                    
                    // Show loading state
                    analyzeBtn.style.display = 'none';
                    loadingIndicator.style.display = 'block';
                    
                    // Note: Form will submit normally, loading state is just for UX
                });
                
                // File input enhancement
                document.getElementById('csvFile').addEventListener('change', function(e) {
                    const file = e.target.files[0];
                    if (file && !file.name.toLowerCase().endsWith('.csv')) {
                        alert('Please select a CSV file from your SEMrush export.');
                        e.target.value = '';
                    }
                });
            </script>
        </body>
    </html>
    """


@app.get("/download-csv/{domain}")
async def download_analysis_csv(domain: str):
    """Download CSV export of analysis results"""
    # For now, return a simple message - we'll implement this with session storage later
    return {"message": "CSV download endpoint ready"}


def generate_csv_content(analysis: RankingIntelligence) -> str:
    """Generate CSV content for download"""
    csv_content = io.StringIO()
    
    # Write header
    csv_content.write("Category,Keyword,Current_Position,Previous_Position,Position_Change,Search_Volume,CPC,Traffic,Traffic_Cost,Opportunity_Score,URL\n")
    
    # Helper function to write keywords to CSV
    def write_keywords_to_csv(keywords, category_name):
        for kw in keywords:
            position_change = kw.position_change if kw.position_change else ""
            previous_pos = kw.previous_position if kw.previous_position else ""
            
            csv_content.write(f'"{category_name}","{kw.keyword}",{kw.current_position},"{previous_pos}","{position_change}",{kw.search_volume},{kw.cpc},{kw.traffic},{kw.traffic_cost},{kw.opportunity_score},"{kw.url}"\n')
    
    # Write all categories
    write_keywords_to_csv(analysis.quick_wins, "Quick Wins")
    write_keywords_to_csv(analysis.content_gaps, "Content Gaps")
    write_keywords_to_csv(analysis.high_value_targets, "High Value Targets")
    write_keywords_to_csv(analysis.declining_keywords, "Declining Keywords")
    write_keywords_to_csv(analysis.traffic_winners, "Traffic Winners")
    write_keywords_to_csv(analysis.serp_feature_opportunities, "SERP Feature Opportunities")
    write_keywords_to_csv(analysis.long_tail_opportunities, "Long Tail Opportunities")
    
    return csv_content.getvalue()


@app.get("/analyze-rankings/")
async def redirect_to_upload():
    """Redirect GET requests to upload form (handles refresh issue)"""
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/upload-form", status_code=302)


@app.post("/analyze-rankings/")
async def analyze_uploaded_rankings(file: UploadFile = File(...)):
    """Upload and analyze SEMrush Organic Research export - Returns beautiful HTML results"""
    
    # Validate file type
    if not file.filename.endswith('.csv'):
        raise HTTPException(
            status_code=400, 
            detail="Please upload a CSV file from SEMrush Organic Research export"
        )
    
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    
    try:
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Analyze the rankings
        result = await intelligence_engine.analyze_ranking_data(temp_path)
        
        # Return beautiful HTML results page with CSV download capability
        return HTMLResponse(content=generate_results_html(result))
        
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


@app.get("/analyze-rankings/json/{filename}")
async def get_analysis_json(filename: str):
    """Download JSON version of analysis results"""
    # This will be implemented later for JSON downloads
    return {"message": "JSON download coming soon!"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_status": "analyzing rankings"}


def generate_results_html(analysis: RankingIntelligence) -> str:
    """Generate beautiful HTML results page"""
    
    # Helper function to format keywords table
    def format_keywords_table(keywords, limit=10):
        if not keywords:
            return "<p class='no-data'>No keywords found in this category</p>"
        
        table_html = """
        <div class="keywords-table-container">
            <table class="keywords-table">
                <thead>
                    <tr>
                        <th>Keyword</th>
                        <th>Position</th>
                        <th>Change</th>
                        <th>Volume</th>
                        <th>CPC</th>
                        <th>Opportunity</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for kw in keywords[:limit]:
            change_icon = ""
            change_class = "neutral"
            if kw.position_change:
                if kw.position_change > 0:
                    change_icon = f"🟢 +{kw.position_change}"
                    change_class = "positive"
                elif kw.position_change < 0:
                    change_icon = f"🔴 {kw.position_change}"
                    change_class = "negative"
            
            table_html += f"""
                <tr>
                    <td class="keyword-cell">
                        <strong>{kw.keyword}</strong>
                        <small>{kw.url}</small>
                    </td>
                    <td class="position-cell">#{kw.current_position}</td>
                    <td class="change-cell {change_class}">{change_icon or '-'}</td>
                    <td class="volume-cell">{kw.search_volume:,}</td>
                    <td class="cpc-cell">${kw.cpc:.2f}</td>
                    <td class="opportunity-cell">
                        <span class="opportunity-score">{kw.opportunity_score:.1f}/10</span>
                    </td>
                </tr>
            """
        
        table_html += """
                </tbody>
            </table>
        </div>
        """
        
        if len(keywords) > limit:
            table_html += f"<p class='more-results'>+ {len(keywords) - limit} more keywords...</p>"
        
        return table_html
    
    # Helper function for insights list
    def format_insights_list(insights):
        if not insights:
            return "<p>No insights available</p>"
        
        html = "<ul class='insights-list'>"
        for insight in insights:
            html += f"<li>{insight}</li>"
        html += "</ul>"
        return html
    
    # Helper function for action items
    def format_action_items(actions):
        if not actions:
            return "<p>No action items available</p>"
        
        html = "<ul class='action-items-list'>"
        for action in actions:
            html += f"<li>{action}</li>"
        html += "</ul>"
        return html

    return f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>🏥 Healthcare SEO Analysis Results - {analysis.site_domain}</title>
            <style>
                :root {{
                    --primary-blue: #2563eb;
                    --primary-blue-dark: #1d4ed8;
                    --healthcare-red: #dc2626;
                    --success-green: #16a34a;
                    --warning-orange: #f59e0b;
                    --light-gray: #f8fafc;
                    --border-gray: #e2e8f0;
                    --text-gray: #475569;
                    --text-dark: #1e293b;
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: var(--text-gray);
                    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
                    min-height: 100vh;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }}
                
                .header {{
                    text-align: center;
                    margin-bottom: 2rem;
                    background: white;
                    padding: 2rem;
                    border-radius: 16px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }}
                
                .header h1 {{
                    color: var(--primary-blue);
                    font-size: 2.2rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                }}
                
                .header .domain {{
                    color: var(--healthcare-red);
                    font-size: 1.3rem;
                    font-weight: 600;
                    margin-bottom: 1rem;
                }}
                
                .overview-stats {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1.5rem;
                    margin-bottom: 3rem;
                }}
                
                .stat-card {{
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
                    text-align: center;
                    transition: transform 0.3s ease;
                }}
                
                .stat-card:hover {{
                    transform: translateY(-4px);
                }}
                
                .stat-card .number {{
                    font-size: 2rem;
                    font-weight: 700;
                    color: var(--primary-blue);
                    display: block;
                }}
                
                .stat-card .label {{
                    color: var(--text-gray);
                    font-size: 0.9rem;
                    margin-top: 0.5rem;
                }}
                
                .opportunities-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
                    gap: 2rem;
                    margin-bottom: 3rem;
                }}
                
                .opportunity-section {{
                    background: white;
                    border-radius: 16px;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                    overflow: hidden;
                }}
                
                .opportunity-header {{
                    padding: 1.5rem;
                    background: linear-gradient(135deg, var(--primary-blue) 0%, var(--primary-blue-dark) 100%);
                    color: white;
                }}
                
                .opportunity-header h3 {{
                    font-size: 1.3rem;
                    margin-bottom: 0.5rem;
                }}
                
                .opportunity-header .count {{
                    font-size: 2rem;
                    font-weight: 700;
                }}
                
                .opportunity-content {{
                    padding: 1.5rem;
                }}
                
                .keywords-table-container {{
                    overflow-x: auto;
                }}
                
                .keywords-table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 1rem 0;
                }}
                
                .keywords-table th {{
                    background: var(--light-gray);
                    padding: 0.75rem;
                    text-align: left;
                    font-weight: 600;
                    color: var(--text-dark);
                    border-bottom: 2px solid var(--border-gray);
                }}
                
                .keywords-table td {{
                    padding: 0.75rem;
                    border-bottom: 1px solid var(--border-gray);
                }}
                
                .keyword-cell {{
                    max-width: 200px;
                }}
                
                .keyword-cell strong {{
                    display: block;
                    color: var(--text-dark);
                    margin-bottom: 0.25rem;
                }}
                
                .keyword-cell small {{
                    color: var(--text-gray);
                    font-size: 0.8rem;
                    opacity: 0.7;
                }}
                
                .position-cell {{
                    font-weight: 600;
                    color: var(--primary-blue);
                }}
                
                .change-cell.positive {{
                    color: var(--success-green);
                    font-weight: 600;
                }}
                
                .change-cell.negative {{
                    color: var(--healthcare-red);
                    font-weight: 600;
                }}
                
                .opportunity-score {{
                    background: linear-gradient(135deg, var(--success-green) 0%, var(--warning-orange) 100%);
                    color: white;
                    padding: 0.25rem 0.5rem;
                    border-radius: 4px;
                    font-weight: 600;
                    font-size: 0.8rem;
                }}
                
                .insights-section {{
                    background: white;
                    border-radius: 16px;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }}
                
                .insights-section h3 {{
                    color: var(--healthcare-red);
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                }}
                
                .insights-list {{
                    list-style: none;
                    margin: 1rem 0;
                }}
                
                .insights-list li {{
                    padding: 0.75rem;
                    margin: 0.5rem 0;
                    background: var(--light-gray);
                    border-radius: 8px;
                    border-left: 4px solid var(--primary-blue);
                }}
                
                .action-items-section {{
                    background: linear-gradient(135deg, var(--healthcare-red) 0%, #b91c1c 100%);
                    color: white;
                    border-radius: 16px;
                    padding: 2rem;
                    margin-bottom: 2rem;
                    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
                }}
                
                .action-items-section h3 {{
                    font-size: 1.5rem;
                    margin-bottom: 1rem;
                }}
                
                .action-items-list {{
                    list-style: none;
                    margin: 1rem 0;
                }}
                
                .action-items-list li {{
                    padding: 0.75rem;
                    margin: 0.5rem 0;
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    border-left: 4px solid white;
                }}
                
                .download-section {{
                    text-align: center;
                    margin: 3rem 0;
                }}
                
                .download-btn {{
                    background: linear-gradient(135deg, var(--success-green) 0%, #15803d 100%);
                    color: white;
                    padding: 1rem 2rem;
                    border: none;
                    border-radius: 8px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    transition: all 0.3s ease;
                    margin: 0 1rem;
                }}
                
                .download-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px -5px rgba(22, 163, 74, 0.4);
                }}
                
                .back-btn {{
                    background: linear-gradient(135deg, var(--text-gray) 0%, #334155 100%);
                    color: white;
                    padding: 1rem 2rem;
                    border: none;
                    border-radius: 8px;
                    font-size: 1.1rem;
                    font-weight: 600;
                    cursor: pointer;
                    text-decoration: none;
                    display: inline-block;
                    transition: all 0.3s ease;
                    margin: 0 1rem;
                }}
                
                .back-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 8px 25px -5px rgba(71, 85, 105, 0.4);
                }}
                
                .no-data {{
                    text-align: center;
                    color: var(--text-gray);
                    font-style: italic;
                    padding: 2rem;
                }}
                
                .more-results {{
                    text-align: center;
                    color: var(--text-gray);
                    font-style: italic;
                    margin-top: 1rem;
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 1rem;
                    }}
                    
                    .opportunities-grid {{
                        grid-template-columns: 1fr;
                    }}
                    
                    .keywords-table {{
                        font-size: 0.8rem;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Healthcare SEO Analysis Results</h1>
                    <p class="domain">{analysis.site_domain}</p>
                    <p>Complete ranking intelligence analysis with actionable opportunities</p>
                </div>
                
                <div class="overview-stats">
                    <div class="stat-card">
                        <span class="number">{analysis.total_keywords:,}</span>
                        <div class="label">Total Keywords</div>
                    </div>
                    <div class="stat-card">
                        <span class="number">{analysis.avg_position:.1f}</span>
                        <div class="label">Average Position</div>
                    </div>
                    <div class="stat-card">
                        <span class="number">{analysis.total_traffic:,}</span>
                        <div class="label">Monthly Traffic</div>
                    </div>
                    <div class="stat-card">
                        <span class="number">${analysis.total_traffic_value:,.0f}</span>
                        <div class="label">Traffic Value</div>
                    </div>
                </div>
                
                <div class="opportunities-grid">
                    <div class="opportunity-section">
                        <div class="opportunity-header">
                            <h3>🎯 Quick Wins</h3>
                            <div class="count">{len(analysis.quick_wins)}</div>
                        </div>
                        <div class="opportunity-content">
                            <p><strong>Keywords in positions 4-10 that can reach top 3</strong></p>
                            {format_keywords_table(analysis.quick_wins)}
                        </div>
                    </div>
                    
                    <div class="opportunity-section">
                        <div class="opportunity-header">
                            <h3>📝 Content Gaps</h3>
                            <div class="count">{len(analysis.content_gaps)}</div>
                        </div>
                        <div class="opportunity-content">
                            <p><strong>Positions 11-20 needing content development</strong></p>
                            {format_keywords_table(analysis.content_gaps)}
                        </div>
                    </div>
                    
                    <div class="opportunity-section">
                        <div class="opportunity-header">
                            <h3>💰 High Value Targets</h3>
                            <div class="count">{len(analysis.high_value_targets)}</div>
                        </div>
                        <div class="opportunity-content">
                            <p><strong>High CPC keywords with revenue potential</strong></p>
                            {format_keywords_table(analysis.high_value_targets)}
                        </div>
                    </div>
                    
                    <div class="opportunity-section">
                        <div class="opportunity-header">
                            <h3>⚠️ Declining Keywords</h3>
                            <div class="count">{len(analysis.declining_keywords)}</div>
                        </div>
                        <div class="opportunity-content">
                            <p><strong>Keywords losing positions - need attention</strong></p>
                            {format_keywords_table(analysis.declining_keywords)}
                        </div>
                    </div>
                </div>
                
                <div class="insights-section">
                    <h3>🧠 Strategic Insights</h3>
                    {format_insights_list(analysis.insights)}
                </div>
                
                <div class="action-items-section">
                    <h3>🚀 Priority Action Items</h3>
                    {format_action_items(analysis.action_items)}
                </div>
                
                <div class="download-section">
                    <a href="/upload-form" class="back-btn">📊 Analyze Another Site</a>
                    <button class="download-btn" onclick="downloadJSON()">💾 Download Full JSON Report</button>
                    <button class="download-btn" onclick="downloadCSV()">📈 Download CSV Report</button>
                </div>
            </div>
            
            <script>
                function downloadJSON() {{
                    const data = {analysis.json()};
                    const blob = new Blob([JSON.stringify(data, null, 2)], {{type: 'application/json'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = '{analysis.site_domain}_seo_analysis.json';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}
                
                function downloadCSV() {{
                    // Generate CSV from analysis data
                    const csvRows = ['Category,Keyword,Position,Volume,CPC,Opportunity'];
                    
                    // Add quick wins
                    const quickWins = {[{"keyword": kw.keyword, "current_position": kw.current_position, "search_volume": kw.search_volume, "cpc": kw.cpc, "opportunity_score": kw.opportunity_score} for kw in analysis.quick_wins[:10]]};
                    quickWins.forEach(kw => {{
                        csvRows.push(`"Quick Wins","${{kw.keyword}}",${{kw.current_position}},${{kw.search_volume}},${{kw.cpc}},${{kw.opportunity_score}}`);
                    }});
                    
                    // Add content gaps
                    const contentGaps = {[{"keyword": kw.keyword, "current_position": kw.current_position, "search_volume": kw.search_volume, "cpc": kw.cpc, "opportunity_score": kw.opportunity_score} for kw in analysis.content_gaps[:10]]};
                    contentGaps.forEach(kw => {{
                        csvRows.push(`"Content Gaps","${{kw.keyword}}",${{kw.current_position}},${{kw.search_volume}},${{kw.cpc}},${{kw.opportunity_score}}`);
                    }});
                    
                    // Add high value
                    const highValue = {[{"keyword": kw.keyword, "current_position": kw.current_position, "search_volume": kw.search_volume, "cpc": kw.cpc, "opportunity_score": kw.opportunity_score} for kw in analysis.high_value_targets[:10]]};
                    highValue.forEach(kw => {{
                        csvRows.push(`"High Value","${{kw.keyword}}",${{kw.current_position}},${{kw.search_volume}},${{kw.cpc}},${{kw.opportunity_score}}`);
                    }});
                    
                    const csvContent = csvRows.join('\\n');
                    const blob = new Blob([csvContent], {{type: 'text/csv;charset=utf-8;'}});
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = '{analysis.site_domain}_seo_opportunities.csv';
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                    URL.revokeObjectURL(url);
                }}
            </script>
        </body>
    </html>
    """


if __name__ == "__main__":
    print("🏥 Starting Healthcare SEO Intelligence Engine...")
    print("📊 Optimized for SEMrush Organic Research exports...")
    uvicorn.run(app, host="0.0.0.0", port=8003)