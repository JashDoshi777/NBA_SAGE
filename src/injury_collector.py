"""
NBA ML Prediction System - Injury Collector
============================================
Real-time injury data integration using nbainjuries package.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, List
import json
import logging

# nbainjuries requires Java/JVM - make import completely optional
HAS_NBA_INJURIES = False
nba_injury = None

try:
    from nbainjuries import injury as nba_injury
    HAS_NBA_INJURIES = True
except Exception as e:
    # Catch ALL exceptions (ImportError, JVMNotFoundException, etc.)
    logging.debug(f"nbainjuries not available: {e}")

from src.config import API_CACHE_DIR, INJURY_IMPACT

logger = logging.getLogger(__name__)

# =============================================================================
# INJURY CACHE
# =============================================================================
class InjuryCache:
    """Cache for injury data with configurable refresh interval."""
    
    def __init__(self, cache_dir: Path = API_CACHE_DIR, cache_hours: float = 3.0):
        self.cache_dir = cache_dir
        self.cache_file = cache_dir / "injury_cache.json"
        self.cache_hours = cache_hours
    
    def is_cache_valid(self) -> bool:
        """Check if cache exists and is within refresh window."""
        if not self.cache_file.exists():
            return False
        
        with open(self.cache_file, 'r') as f:
            cache_data = json.load(f)
        
        cached_time = datetime.fromisoformat(cache_data.get("timestamp", "2000-01-01"))
        return datetime.now() - cached_time < timedelta(hours=self.cache_hours)
    
    def load(self) -> Optional[Dict]:
        """Load cached injury data."""
        if not self.cache_file.exists():
            return None
        
        with open(self.cache_file, 'r') as f:
            return json.load(f)
    
    def save(self, data: Dict):
        """Save injury data to cache."""
        cache_data = {
            "timestamp": datetime.now().isoformat(),
            "injuries": data
        }
        with open(self.cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)


# =============================================================================
# INJURY COLLECTOR
# =============================================================================
class InjuryCollector:
    """Collects and processes injury data for predictions."""
    
    def __init__(self, cache_hours: float = 3.0):
        self.cache = InjuryCache(cache_hours=cache_hours)
        self._injuries_df = None
        self._last_fetch_time = None
        self._memory_cache_ttl = 300  # 5 minutes in-memory cache
    
    def fetch_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch current injury data from NBA injury reports.
        
        Returns DataFrame with columns:
        - Player: Player's full name
        - Team: Team name
        - Status: Out, Questionable, Doubtful, Probable, Available
        - Reason: Injury description
        """
        # Use in-memory cache if fresh (for fast repeated calls)
        if not force_refresh and self._injuries_df is not None and self._last_fetch_time:
            elapsed = (datetime.now() - self._last_fetch_time).total_seconds()
            if elapsed < self._memory_cache_ttl:
                return self._injuries_df
        
        # Check file cache next
        if not force_refresh and self.cache.is_cache_valid():
            cache_data = self.cache.load()
            if cache_data and "injuries" in cache_data:
                self._injuries_df = pd.DataFrame(cache_data["injuries"])
                self._last_fetch_time = datetime.now()
                return self._injuries_df
        
        # Fetch fresh data from nbainjuries API
        if HAS_NBA_INJURIES:
            try:
                # NBA injury reports are published at 30-min intervals (e.g., 5:00 PM, 5:30 PM ET)
                # Try multiple recent timestamps to find the latest available report
                now = datetime.now()
                
                for hours_ago in [0, 1, 2, 3, 6, 12]:
                    for minutes in [0, 30]:
                        try:
                            timestamp = now.replace(minute=minutes, second=0, microsecond=0) - timedelta(hours=hours_ago)
                            
                            # Check if report exists before trying to fetch
                            if nba_injury.check_reportvalid(timestamp):
                                df = nba_injury.get_reportdata(timestamp, return_df=True)
                                
                                if df is not None and not df.empty:
                                    # Standardize column names for our use
                                    df = self._standardize_columns(df)
                                    logger.info(f"Successfully fetched {len(df)} injury records from NBA report ({timestamp})")
                                    self.cache.save(df.to_dict('records'))
                                    self._injuries_df = df
                                    self._last_fetch_time = datetime.now()
                                    return df
                        except Exception:
                            continue
                
                logger.warning("No valid injury report found in recent timestamps")
                
            except Exception as e:
                logger.warning(f"Failed to fetch injuries from nbainjuries API: {e}")
        else:
            logger.debug("nbainjuries package not installed, using empty injury data")
        
        # Return empty DataFrame
        self._injuries_df = self._get_empty_injuries_df()
        self._last_fetch_time = datetime.now()
        return self._injuries_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names from nbainjuries format to our format."""
        # Map nbainjuries columns to our expected format
        column_mapping = {
            'Player': 'player_name',
            'Team': 'team', 
            'Status': 'status',
            'Reason': 'injury',
            'Game Date': 'date'
        }
        
        # Rename columns that exist
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        return df
    
    def _get_empty_injuries_df(self) -> pd.DataFrame:
        """Return empty injuries DataFrame with proper schema."""
        return pd.DataFrame(columns=[
            "player_name", "team", "status", "injury", "date"
        ])
    
    def get_team_injuries(self, team_abbrev: str) -> pd.DataFrame:
        """Get injuries for a specific team."""
        df = self.fetch_injuries()
        if df.empty:
            return df
        return df[df["team"] == team_abbrev]
    
    def calculate_injury_impact(self, team_abbrev: str, 
                                 player_usage: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate total injury impact for a team.
        
        Args:
            team_abbrev: Team abbreviation (e.g., "LAL")
            player_usage: Dict mapping player names to usage rates (0-1)
                         If None, uses equal weighting
        
        Returns:
            Injury impact score (0 = no injuries, higher = more impact)
        """
        team_injuries = self.get_team_injuries(team_abbrev)
        
        if team_injuries.empty:
            return 0.0
        
        total_impact = 0.0
        for _, injury in team_injuries.iterrows():
            status = injury.get("status", "Available")
            base_impact = INJURY_IMPACT.get(status, 0.0)
            
            # Weight by player usage if provided
            if player_usage and injury["player_name"] in player_usage:
                player_weight = player_usage[injury["player_name"]]
            else:
                # Default: assume equal importance for all injured players
                player_weight = 0.2
            
            total_impact += base_impact * player_weight
        
        return min(total_impact, 1.0)  # Cap at 1.0
    
    def get_injury_summary(self, team_abbrev: str) -> Dict:
        """Get a summary of team injuries for display."""
        team_injuries = self.get_team_injuries(team_abbrev)
        
        summary = {
            "total_injuries": len(team_injuries),
            "out": 0,
            "questionable": 0,
            "doubtful": 0,
            "probable": 0,
            "players": []
        }
        
        if team_injuries.empty:
            return summary
        
        for _, injury in team_injuries.iterrows():
            status = injury.get("status", "").lower()
            if "out" in status:
                summary["out"] += 1
            elif "questionable" in status:
                summary["questionable"] += 1
            elif "doubtful" in status:
                summary["doubtful"] += 1
            elif "probable" in status:
                summary["probable"] += 1
            
            summary["players"].append({
                "name": injury.get("player_name", "Unknown"),
                "status": injury.get("status", "Unknown"),
                "injury": injury.get("injury", "Unknown")
            })
        
        return summary


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    collector = InjuryCollector()
    
    print("Fetching current NBA injuries...")
    injuries_df = collector.fetch_injuries()
    
    if not injuries_df.empty:
        print(f"\nFound {len(injuries_df)} injury reports")
        print("\nSample injuries:")
        print(injuries_df.head(10))
        
        # Test impact calculation
        print("\nInjury impact for LAL:", collector.calculate_injury_impact("LAL"))
    else:
        print("No injury data available")
