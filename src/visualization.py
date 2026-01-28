"""
NBA ML Prediction System - Data Visualization
==============================================
Create and save visualizations for analysis and reporting.
All graphs are saved to the 'graphs' folder.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from typing import Optional, List, Dict
from datetime import datetime
import logging

from src.config import PROJECT_ROOT, RAW_DATA_DIR, PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)

# =============================================================================
# CONFIGURATION
# =============================================================================
GRAPHS_DIR = PROJECT_ROOT / "graphs"
GRAPHS_DIR.mkdir(exist_ok=True)

# Style settings
plt.style.use('dark_background')
COLORS = {
    'primary': '#7c3aed',
    'secondary': '#00d4ff',
    'success': '#10b981',
    'warning': '#f59e0b',
    'danger': '#ef4444',
    'gradient': ['#7c3aed', '#00d4ff', '#f472b6']
}

def save_figure(fig, name: str, dpi: int = 150):
    """Save figure to graphs folder."""
    path = GRAPHS_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='#1a1a2e', edgecolor='none')
    logger.info(f"Saved graph to {path}")
    plt.close(fig)
    return path


# =============================================================================
# TEAM PERFORMANCE VISUALIZATIONS
# =============================================================================
class TeamVisualizer:
    """Visualization for team-level statistics."""
    
    def plot_elo_history(self, elo_history: pd.DataFrame, team_abbrev: str = None) -> Path:
        """
        Plot ELO rating history over time.
        
        Args:
            elo_history: DataFrame with columns [date, team, elo]
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        if team_abbrev:
            data = elo_history[elo_history['team'] == team_abbrev]
            ax.plot(data['date'], data['elo'], color=COLORS['primary'], linewidth=2)
            ax.fill_between(data['date'], 1500, data['elo'], alpha=0.3, color=COLORS['primary'])
            title = f"{team_abbrev} ELO Rating History"
        else:
            # Plot top 5 teams
            for i, (team, data) in enumerate(elo_history.groupby('team')):
                if i < 5:
                    ax.plot(data['date'], data['elo'], label=team, linewidth=2)
            ax.legend(loc='upper left')
            title = "ELO Rating History - Top Teams"
        
        ax.axhline(y=1500, color='white', linestyle='--', alpha=0.3, label='Average')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('ELO Rating', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        return save_figure(fig, f"elo_history_{team_abbrev or 'all'}")
    
    def plot_team_comparison(self, team_stats: pd.DataFrame, 
                              metrics: List[str] = None) -> Path:
        """
        Radar chart comparing multiple teams.
        """
        if metrics is None:
            metrics = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'FG_PCT']
        
        # Normalize metrics to 0-1 scale
        normalized = team_stats[metrics].copy()
        for col in metrics:
            normalized[col] = (normalized[col] - normalized[col].min()) / (normalized[col].max() - normalized[col].min())
        
        # Create radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]
        
        for i, (idx, row) in enumerate(team_stats.head(5).iterrows()):
            values = normalized.loc[idx, metrics].tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, label=row.get('TEAM_ABBREVIATION', f'Team {i+1}'))
            ax.fill(angles, values, alpha=0.1)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
        ax.set_title('Team Comparison', fontsize=16, fontweight='bold', pad=20)
        
        return save_figure(fig, "team_comparison_radar")
    
    def plot_standings(self, standings: pd.DataFrame) -> Path:
        """
        Horizontal bar chart of team standings by win percentage.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        data = standings.sort_values('W_PCT', ascending=True).tail(15)
        colors = [COLORS['primary'] if i >= 7 else COLORS['secondary'] 
                  for i in range(len(data))]
        
        bars = ax.barh(data['TEAM_ABBREVIATION'], data['W_PCT'], color=colors, edgecolor='white', linewidth=0.5)
        
        # Add playoff line
        ax.axvline(x=0.5, color=COLORS['warning'], linestyle='--', linewidth=2, label='Playoff Cutoff')
        
        ax.set_xlabel('Win Percentage', fontsize=12)
        ax.set_title('Team Standings', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.legend()
        ax.grid(axis='x', alpha=0.2)
        
        # Add value labels
        for bar, val in zip(bars, data['W_PCT']):
            ax.text(val + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1%}', va='center', fontsize=10)
        
        return save_figure(fig, "standings")


# =============================================================================
# GAME PREDICTION VISUALIZATIONS
# =============================================================================
class GameVisualizer:
    """Visualization for game predictions and analysis."""
    
    def plot_prediction_calibration(self, predictions: pd.DataFrame) -> Path:
        """
        Calibration curve - how well do probabilities match actual outcomes.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Bin predictions
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        predicted_proba = predictions['predicted_proba']
        actual = predictions['actual']
        
        bin_indices = np.digitize(predicted_proba, bins) - 1
        bin_indices = np.clip(bin_indices, 0, 9)
        
        actual_fractions = []
        for i in range(10):
            mask = bin_indices == i
            if mask.sum() > 0:
                actual_fractions.append(actual[mask].mean())
            else:
                actual_fractions.append(np.nan)
        
        # Perfect calibration line
        ax.plot([0, 1], [0, 1], 'w--', linewidth=2, label='Perfect Calibration')
        
        # Actual calibration
        ax.plot(bin_centers, actual_fractions, 'o-', color=COLORS['primary'], 
               linewidth=3, markersize=10, label='Model Calibration')
        
        ax.set_xlabel('Predicted Probability', fontsize=12)
        ax.set_ylabel('Actual Win Rate', fontsize=12)
        ax.set_title('Prediction Calibration Curve', fontsize=16, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        
        return save_figure(fig, "calibration_curve")
    
    def plot_accuracy_by_confidence(self, predictions: pd.DataFrame) -> Path:
        """
        How does accuracy change with prediction confidence?
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate confidence as distance from 0.5
        predictions = predictions.copy()
        predictions['confidence'] = abs(predictions['predicted_proba'] - 0.5)
        predictions['correct'] = predictions['predicted'] == predictions['actual']
        
        # Bin by confidence
        bins = np.linspace(0, 0.5, 6)
        predictions['confidence_bin'] = pd.cut(predictions['confidence'], bins)
        
        accuracy_by_conf = predictions.groupby('confidence_bin')['correct'].mean()
        count_by_conf = predictions.groupby('confidence_bin').size()
        
        # Accuracy plot
        ax1.bar(range(len(accuracy_by_conf)), accuracy_by_conf.values, 
               color=COLORS['primary'], edgecolor='white')
        ax1.set_xticks(range(len(accuracy_by_conf)))
        ax1.set_xticklabels(['Low', 'Med-Low', 'Medium', 'Med-High', 'High'], rotation=45)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_title('Accuracy by Confidence Level', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1)
        ax1.axhline(y=0.5, color='white', linestyle='--', alpha=0.3)
        
        # Count plot
        ax2.bar(range(len(count_by_conf)), count_by_conf.values,
               color=COLORS['secondary'], edgecolor='white')
        ax2.set_xticks(range(len(count_by_conf)))
        ax2.set_xticklabels(['Low', 'Med-Low', 'Medium', 'Med-High', 'High'], rotation=45)
        ax2.set_ylabel('Number of Predictions', fontsize=12)
        ax2.set_title('Prediction Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return save_figure(fig, "accuracy_by_confidence")
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 15) -> Path:
        """
        Bar chart of feature importance.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = importance_df.head(top_n).sort_values('avg_importance', ascending=True)
        
        bars = ax.barh(data['feature'], data['avg_importance'], 
                      color=COLORS['primary'], edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title('Top Features for Game Prediction', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        
        return save_figure(fig, "feature_importance")


# =============================================================================
# MVP VISUALIZATIONS
# =============================================================================
class MVPVisualizer:
    """Visualization for MVP race analysis."""
    
    def plot_mvp_race(self, mvp_df: pd.DataFrame) -> Path:
        """
        Horizontal bar chart of MVP race standings.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        data = mvp_df.head(10).sort_values('mvp_score', ascending=True)
        colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(data)))
        
        bars = ax.barh(data['PLAYER_NAME'], data['mvp_score'], color=colors, edgecolor='white')
        
        ax.set_xlabel('MVP Score', fontsize=12)
        ax.set_title('MVP Race 2024-25', fontsize=16, fontweight='bold')
        ax.grid(axis='x', alpha=0.2)
        
        # Add value labels
        for bar, val in zip(bars, data['mvp_score']):
            ax.text(val + 0.5, bar.get_y() + bar.get_height()/2, 
                   f'{val:.1f}', va='center', fontsize=10)
        
        return save_figure(fig, "mvp_race")
    
    def plot_mvp_similarity(self, mvp_df: pd.DataFrame) -> Path:
        """
        Scatter plot of MVP score vs historical similarity.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(mvp_df['mvp_similarity'], mvp_df['mvp_score'],
                            s=mvp_df['PTS'] * 10, c=mvp_df['mvp_score'],
                            cmap='Purples', alpha=0.7, edgecolor='white')
        
        # Add labels for top candidates
        for idx, row in mvp_df.head(5).iterrows():
            ax.annotate(row['PLAYER_NAME'], 
                       (row['mvp_similarity'], row['mvp_score']),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, color='white')
        
        ax.set_xlabel('Similarity to Historical MVPs', fontsize=12)
        ax.set_ylabel('MVP Score', fontsize=12)
        ax.set_title('MVP Score vs Historical Similarity', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('MVP Score', fontsize=10)
        
        return save_figure(fig, "mvp_similarity_scatter")
    
    def plot_stat_comparison(self, mvp_df: pd.DataFrame, 
                              stats: List[str] = None) -> Path:
        """
        Bar chart comparing stats of top MVP candidates.
        """
        if stats is None:
            stats = ['PTS', 'REB', 'AST']
        
        fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 6))
        if len(stats) == 1:
            axes = [axes]
        
        top_players = mvp_df.head(5)
        
        for ax, stat in zip(axes, stats):
            colors = plt.cm.Purples(np.linspace(0.4, 0.9, len(top_players)))
            bars = ax.bar(top_players['PLAYER_NAME'], top_players[stat], color=colors, edgecolor='white')
            ax.set_ylabel(stat, fontsize=12)
            ax.set_title(f'{stat} Comparison', fontsize=14, fontweight='bold')
            ax.tick_params(axis='x', rotation=45)
            
            for bar, val in zip(bars, top_players[stat]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                       f'{val:.1f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        return save_figure(fig, "mvp_stat_comparison")


# =============================================================================
# CHAMPIONSHIP VISUALIZATIONS
# =============================================================================
class ChampionshipVisualizer:
    """Visualization for championship predictions."""
    
    def plot_championship_odds(self, champ_df: pd.DataFrame) -> Path:
        """
        Pie chart of championship probabilities.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        data = champ_df.head(8)
        colors = plt.cm.Purples(np.linspace(0.3, 0.9, len(data)))
        
        wedges, texts, autotexts = ax.pie(
            data['champ_probability'], 
            labels=data['TEAM_ABBREVIATION'],
            autopct='%1.1f%%',
            colors=colors,
            explode=[0.05] * len(data),
            shadow=True,
            startangle=90
        )
        
        for text in texts:
            text.set_fontsize(12)
            text.set_color('white')
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_color('white')
        
        ax.set_title('Championship Probabilities', fontsize=16, fontweight='bold')
        
        return save_figure(fig, "championship_odds_pie")
    
    def plot_strength_vs_experience(self, champ_df: pd.DataFrame) -> Path:
        """
        Scatter plot of team strength vs playoff experience.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        scatter = ax.scatter(
            champ_df['playoff_experience'],
            champ_df['strength_rating'],
            s=champ_df['champ_probability'] * 3000,
            c=champ_df['champ_probability'],
            cmap='Purples',
            alpha=0.7,
            edgecolor='white',
            linewidth=2
        )
        
        # Add labels
        for idx, row in champ_df.iterrows():
            ax.annotate(
                row['TEAM_ABBREVIATION'],
                (row['playoff_experience'], row['strength_rating']),
                xytext=(10, 5), textcoords='offset points',
                fontsize=11, color='white', fontweight='bold'
            )
        
        ax.set_xlabel('Playoff Experience Index', fontsize=12)
        ax.set_ylabel('Strength Rating', fontsize=12)
        ax.set_title('Championship Contenders: Strength vs Experience', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.2)
        
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Championship Probability', fontsize=10)
        
        return save_figure(fig, "strength_vs_experience")


# =============================================================================
# SEASON ANALYSIS VISUALIZATIONS
# =============================================================================
class SeasonVisualizer:
    """Visualization for historical season analysis."""
    
    def plot_scoring_trends(self, season_data: pd.DataFrame) -> Path:
        """
        Line chart of scoring trends across seasons.
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        ax.plot(season_data['season'], season_data['avg_pts'], 
               color=COLORS['primary'], linewidth=3, marker='o', markersize=8)
        ax.fill_between(season_data['season'], season_data['avg_pts'], alpha=0.3, color=COLORS['primary'])
        
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('Average Points Per Game', fontsize=12)
        ax.set_title('NBA Scoring Trends Over Time', fontsize=16, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.2)
        
        return save_figure(fig, "scoring_trends")
    
    def plot_three_point_revolution(self, season_data: pd.DataFrame) -> Path:
        """
        Dual-axis chart showing 3PA and 3P% trends.
        """
        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax2 = ax1.twinx()
        
        ax1.bar(season_data['season'], season_data['avg_3pa'], 
               color=COLORS['secondary'], alpha=0.7, label='3-Point Attempts')
        ax2.plot(season_data['season'], season_data['avg_3pct'], 
                color=COLORS['primary'], linewidth=3, marker='o', label='3-Point %')
        
        ax1.set_xlabel('Season', fontsize=12)
        ax1.set_ylabel('3-Point Attempts', fontsize=12, color=COLORS['secondary'])
        ax2.set_ylabel('3-Point Percentage', fontsize=12, color=COLORS['primary'])
        ax1.set_title('The 3-Point Revolution', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        return save_figure(fig, "three_point_revolution")


# =============================================================================
# MASTER VISUALIZER
# =============================================================================
class NBAVisualizer:
    """Master class combining all visualization capabilities."""
    
    def __init__(self):
        self.team = TeamVisualizer()
        self.game = GameVisualizer()
        self.mvp = MVPVisualizer()
        self.championship = ChampionshipVisualizer()
        self.season = SeasonVisualizer()
    
    def generate_all_visualizations(self, data: Dict[str, pd.DataFrame]) -> List[Path]:
        """
        Generate all available visualizations from provided data.
        
        Args:
            data: Dict with keys like 'standings', 'mvp', 'championship', etc.
        
        Returns:
            List of paths to saved graphs
        """
        saved_paths = []
        
        if 'standings' in data:
            saved_paths.append(self.team.plot_standings(data['standings']))
        
        if 'mvp' in data:
            saved_paths.append(self.mvp.plot_mvp_race(data['mvp']))
            saved_paths.append(self.mvp.plot_stat_comparison(data['mvp']))
        
        if 'championship' in data:
            saved_paths.append(self.championship.plot_championship_odds(data['championship']))
            saved_paths.append(self.championship.plot_strength_vs_experience(data['championship']))
        
        if 'predictions' in data:
            saved_paths.append(self.game.plot_calibration(data['predictions']))
            saved_paths.append(self.game.plot_accuracy_by_confidence(data['predictions']))
        
        logger.info(f"Generated {len(saved_paths)} visualizations")
        return saved_paths


# =============================================================================
# CLI INTERFACE
# =============================================================================
if __name__ == "__main__":
    print(f"Generating sample visualizations to {GRAPHS_DIR}...")
    
    # Create sample data for testing
    sample_mvp = pd.DataFrame({
        'PLAYER_NAME': ['Shai Gilgeous-Alexander', 'Nikola Jokic', 'Jayson Tatum', 
                       'Luka Doncic', 'Giannis Antetokounmpo'],
        'PTS': [31.5, 26.8, 27.2, 28.5, 30.5],
        'REB': [5.5, 12.5, 8.2, 8.8, 11.5],
        'AST': [6.0, 9.2, 4.8, 8.2, 6.5],
        'mvp_score': [85.2, 82.1, 78.5, 77.2, 76.8],
        'mvp_similarity': [0.92, 0.95, 0.85, 0.88, 0.90]
    })
    
    sample_champ = pd.DataFrame({
        'TEAM_ABBREVIATION': ['OKC', 'CLE', 'BOS', 'DEN', 'MEM', 'HOU', 'NYK', 'GSW'],
        'W_PCT': [0.70, 0.68, 0.65, 0.62, 0.60, 0.58, 0.55, 0.52],
        'playoff_experience': [0.3, 0.5, 0.8, 0.9, 0.4, 0.2, 0.5, 0.95],
        'strength_rating': [45, 42, 40, 38, 35, 33, 30, 28],
        'champ_probability': [0.18, 0.15, 0.14, 0.12, 0.10, 0.09, 0.08, 0.07]
    })
    
    viz = NBAVisualizer()
    
    # Generate sample visualizations
    print("Creating MVP race chart...")
    viz.mvp.plot_mvp_race(sample_mvp)
    
    print("Creating MVP stat comparison...")
    viz.mvp.plot_stat_comparison(sample_mvp)
    
    print("Creating championship odds pie chart...")
    viz.championship.plot_championship_odds(sample_champ)
    
    print("Creating strength vs experience chart...")
    viz.championship.plot_strength_vs_experience(sample_champ)
    
    print(f"\n✅ Visualizations saved to: {GRAPHS_DIR}")
