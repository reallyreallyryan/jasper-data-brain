"""
Desert Spine & Sports - Ranking Intelligence Engine
Specialized for SEMrush Organic Research exports
Analyzes 4,912+ keywords to find ranking opportunities and traffic growth potential
"""

import pandas as pd
import numpy as np
import json
import asyncio
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn


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


class DesertSpineRankingIntelligence:
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
            df = df.dropna(subset=['Keyword', 'Position', 'Search Volume'])
            df['Position'] = pd.to_numeric(df['Position'], errors='coerce')
            df['Search Volume'] = pd.to_numeric(df['Search Volume'], errors='coerce')
            df['Keyword Difficulty'] = pd.to_numeric(df['Keyword Difficulty'], errors='coerce')
            df['CPC'] = pd.to_numeric(df['CPC'], errors='coerce')
            df['Traffic'] = pd.to_numeric(df['Traffic'], errors='coerce')
            df['Traffic (%)'] = pd.to_numeric(df['Traffic (%)'], errors='coerce')
            df['Traffic Cost'] = pd.to_numeric(df['Traffic Cost'], errors='coerce')
            
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
    title="Desert Spine & Sports - Ranking Intelligence Engine",
    description="Specialized SEO ranking analysis for healthcare websites using SEMrush Organic Research exports",
    version="1.0.0"
)

intelligence_engine = DesertSpineRankingIntelligence()


@app.get("/")
async def root():
    return {
        "message": "🏥 Desert Spine & Sports Ranking Intelligence Engine",
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
    """Upload form for SEMrush Organic Research exports"""
    return """
    <html>
        <head>
            <title>🏥 Desert Spine & Sports - Ranking Intelligence</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 900px; margin: 50px auto; padding: 20px; }
                .upload-area { border: 2px dashed #007acc; padding: 40px; text-align: center; margin: 20px 0; background: #f8f9fa; }
                button { background: #007acc; color: white; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
                button:hover { background: #005999; }
                .stats { display: flex; justify-content: space-around; margin: 20px 0; }
                .stat-box { background: #e9ecef; padding: 15px; border-radius: 4px; text-align: center; }
                .healthcare { color: #dc3545; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1>🏥 Desert Spine & Sports - Ranking Intelligence Engine</h1>
            <p class="healthcare">Specialized for healthcare SEO & spine/sports medicine practices</p>
            
            <form action="/analyze-rankings/" method="post" enctype="multipart/form-data">
                <div class="upload-area">
                    <h3>📊 Upload SEMrush Organic Research Export</h3>
                    <input type="file" name="file" accept=".csv" required>
                    <br><br>
                    <p>Expected format: <strong>SEMrush → Organic Research → Export Organic Keywords</strong></p>
                    <button type="submit">🚀 Analyze Rankings</button>
                </div>
            </form>
            
            <div class="stats">
                <div class="stat-box">
                    <h4>🎯 Quick Wins</h4>
                    <p>Keywords in positions 4-10 that can reach top 3</p>
                </div>
                <div class="stat-box">
                    <h4>📝 Content Gaps</h4>
                    <p>Positions 11-20 needing content optimization</p>
                </div>
                <div class="stat-box">
                    <h4>💰 High Value</h4>
                    <p>High CPC keywords with revenue potential</p>
                </div>
                <div class="stat-box">
                    <h4>⚠️ Declining</h4>
                    <p>Keywords losing positions that need attention</p>
                </div>
            </div>
            
            <h3>🏥 Healthcare SEO Features:</h3>
            <ul>
                <li><strong>Medical Keyword Analysis</strong> - Spine, sports medicine, pain treatment terms</li>
                <li><strong>Patient Intent Classification</strong> - Information vs appointment seeking</li>
                <li><strong>Local SEO Opportunities</strong> - "Near me" and location-based terms</li>
                <li><strong>Treatment Page Optimization</strong> - Service-specific ranking opportunities</li>
                <li><strong>Competition Analysis</strong> - vs other medical practices</li>
            </ul>
        </body>
    </html>
    """


@app.post("/analyze-rankings/", response_model=RankingIntelligence)
async def analyze_uploaded_rankings(file: UploadFile = File(...)):
    """Upload and analyze SEMrush Organic Research export"""
    
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
        
        return result
        
    finally:
        # Clean up temp file
        if Path(temp_path).exists():
            Path(temp_path).unlink()


@app.get("/health")
async def health_check():
    return {"status": "healthy", "engine_status": "analyzing rankings"}


if __name__ == "__main__":
    print("🏥 Starting Desert Spine & Sports Ranking Intelligence Engine...")
    print("📊 Optimized for SEMrush Organic Research exports...")
    uvicorn.run(app, host="0.0.0.0", port=8003)