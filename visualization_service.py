import matplotlib

matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import io
import base64
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VisualizationService:
    """
    Intelligent data visualization service that:
    - Analyzes query results
    - Detects best chart type
    - Generates publication-quality visualizations
    - Returns base64 encoded images for web display
    """

    def __init__(self):
        """Initialize visualization service with styling."""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        self.colors = {
            'primary': '#1f77b4',
            'success': '#2ca02c',
            'warning': '#ff7f0e',
            'danger': '#d62728',
            'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        }

    def analyze_data_for_viz(self, df: pd.DataFrame) -> dict:
        """
        Analyze dataframe and suggest visualization type.

        Returns:
        {
            'recommended_chart': 'bar|line|pie|scatter|histogram|heatmap|boxplot',
            'reason': 'explanation',
            'x_column': 'column_name',
            'y_column': 'column_name',
            'can_visualize': True/False
        }
        """
        if df is None or df.empty:
            return {
                'can_visualize': False,
                'reason': 'No data to visualize',
                'recommended_chart': None
            }

        # --- FIX: (Issue 12) Sample large datasets instead of failing ---
        MAX_VIZ_ROWS = 10000
        if len(df) > MAX_VIZ_ROWS:
            logger.info(f"Dataset too large ({len(df)} rows). Sampling to {MAX_VIZ_ROWS} rows for visualization.")
            df = df.sample(n=MAX_VIZ_ROWS, random_state=42)
        # --- END FIX ---

        # Get column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = self._detect_date_columns(df)

        # Rule 1: Time series data (date + numeric)
        if date_cols and numeric_cols:
            return {
                'can_visualize': True,
                'recommended_chart': 'line',
                'reason': f'Time series data detected with {len(date_cols)} date column(s) and {len(numeric_cols)} numeric column(s)',
                'x_column': date_cols[0],
                'y_column': numeric_cols[0]
            }

        # Rule 2: Categorical + Numeric (bar chart)
        if categorical_cols and numeric_cols:
            unique_categories = df[categorical_cols[0]].nunique()
            MAX_BAR_CATEGORIES = 20
            if unique_categories <= MAX_BAR_CATEGORIES:
                return {
                    'can_visualize': True,
                    'recommended_chart': 'bar',
                    'reason': f'Categorical data ({unique_categories} categories) with numeric values - ideal for comparison',
                    'x_column': categorical_cols[0],
                    'y_column': numeric_cols[0]
                }
            else:
                return {
                    'can_visualize': True,
                    'recommended_chart': 'histogram',
                    'reason': f'Too many categories ({unique_categories}) - showing distribution instead',
                    'x_column': numeric_cols[0],
                    'y_column': None
                }

        # Rule 3: Percentage/proportion data (pie chart)
        if self._has_percentage_data(df, categorical_cols, numeric_cols):
            return {
                'can_visualize': True,
                'recommended_chart': 'pie',
                'reason': 'Data represents proportions or percentages',
                'x_column': categorical_cols[0] if categorical_cols else None,
                'y_column': numeric_cols[0] if numeric_cols else None
            }

        # Rule 4: Two numeric columns (scatter plot)
        if len(numeric_cols) == 2:
            return {
                'can_visualize': True,
                'recommended_chart': 'scatter',
                'reason': 'Two numeric columns detected - showing correlation',
                'x_column': numeric_cols[0],
                'y_column': numeric_cols[1]
            }

        # Rule 5: Single numeric column (histogram)
        if len(numeric_cols) == 1:
            return {
                'can_visualize': True,
                'recommended_chart': 'histogram',
                'reason': 'Single numeric column - showing distribution',
                'x_column': numeric_cols[0],
                'y_column': None
            }

        # Rule 6: Multiple numeric columns (heatmap or boxplot)
        if len(numeric_cols) >= 3:
            MIN_ROWS_FOR_HEATMAP = 10
            if len(df) >= MIN_ROWS_FOR_HEATMAP:  # Need enough data for correlation
                return {
                    'can_visualize': True,
                    'recommended_chart': 'heatmap',
                    'reason': f'{len(numeric_cols)} numeric columns detected - showing correlations',
                    'x_column': None,
                    'y_column': None
                }
            else:
                return {
                    'can_visualize': True,
                    'recommended_chart': 'boxplot',
                    'reason': f'{len(numeric_cols)} numeric columns - showing statistical distributions',
                    'x_column': None,
                    'y_column': None
                }

        # Default: If we have any numeric data, try bar chart
        if numeric_cols:
            return {
                'can_visualize': True,
                'recommended_chart': 'bar',
                'reason': 'Default visualization for numeric data',
                'x_column': df.columns[0],
                'y_column': numeric_cols[0]
            }

        return {
            'can_visualize': False,
            'reason': 'No suitable numeric data found for visualization',
            'recommended_chart': None
        }

    def create_chart(self, df: pd.DataFrame, chart_type: str, **kwargs) -> str:
        """
        Generate chart and return as base64 encoded PNG.

        Args:
            df: Pandas dataframe with query results
            chart_type: Type of chart to create
            **kwargs: Additional parameters (title, colors, etc.)

        Returns:
            Base64 encoded PNG image string
        """
        try:
            if chart_type == 'line':
                return self.create_line_chart(df, kwargs.get('x'), kwargs.get('y'), kwargs.get('title', 'Line Chart'))
            elif chart_type == 'bar':
                return self.create_bar_chart(df, kwargs.get('x'), kwargs.get('y'), kwargs.get('title', 'Bar Chart'))
            elif chart_type == 'pie':
                return self.create_pie_chart(df, kwargs.get('x'), kwargs.get('y'), kwargs.get('title', 'Pie Chart'))
            elif chart_type == 'scatter':
                return self.create_scatter_plot(df, kwargs.get('x'), kwargs.get('y'),
                                                kwargs.get('title', 'Scatter Plot'))
            elif chart_type == 'histogram':
                return self.create_histogram(df, kwargs.get('x'), kwargs.get('bins', 30),
                                             kwargs.get('title', 'Histogram'))
            elif chart_type == 'heatmap':
                return self.create_heatmap(df, kwargs.get('title', 'Correlation Heatmap'))
            elif chart_type == 'boxplot':
                return self.create_box_plot(df, None, kwargs.get('title', 'Box Plot'))
            else:
                raise ValueError(f"Unsupported chart type: {chart_type}")
        except Exception as e:
            logger.error(f"Error creating {chart_type} chart: {str(e)}", exc_info=True)
            raise Exception(f"Error creating {chart_type} chart: {str(e)}")

    def create_line_chart(self, df: pd.DataFrame, x: str, y: str, title: str) -> str:
        """Create time series or trend line chart"""
        fig, ax = self._setup_chart(title, x or 'X', y or 'Y')

        # Auto-select columns if not specified
        if not x or not y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(df.columns) >= 2:
                x = df.columns[0]
                y = numeric_cols[0] if numeric_cols else df.columns[1]

        # Handle date columns
        if pd.api.types.is_object_dtype(df[x]):
            try:
                df[x] = pd.to_datetime(df[x])
            except:
                pass

        ax.plot(df[x], df[y], marker='o', linewidth=2, markersize=6, color=self.colors['primary'])

        # Rotate x-axis labels if needed
        if len(df) > 10:
            plt.xticks(rotation=45, ha='right')

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)

        return self._fig_to_base64(fig)

    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, title: str) -> str:
        """Create categorical bar chart with truncation notice."""
        original_count = len(df)

        # Auto-select columns if not specified
        if not x or not y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            x = categorical_cols[0] if categorical_cols else df.columns[0]
            y = numeric_cols[0] if numeric_cols else df.columns[1]

        # Limit to top 20 categories if too many
        MAX_BAR_CATEGORIES = 20
        if df[x].nunique() > MAX_BAR_CATEGORIES:
            df = df.nlargest(MAX_BAR_CATEGORIES, y)
            title = f"{title} (Top {MAX_BAR_CATEGORIES} of {original_count} categories)"

        # Sort by value
        df_sorted = df.sort_values(by=y, ascending=True)

        fig, ax = self._setup_chart(title, x or 'Category', y or 'Value')

        # Horizontal bar chart for better label visibility
        ax.barh(df_sorted[x].astype(str), df_sorted[y], color=self.colors['palette'])
        ax.set_xlabel(y, fontsize=12)
        ax.set_ylabel(x, fontsize=12)

        # Add value labels
        for i, v in enumerate(df_sorted[y]):
            ax.text(v, i, f' {v:,.0f}', va='center', fontsize=9)

        return self._fig_to_base64(fig)

    def create_pie_chart(self, df: pd.DataFrame, labels: str, values: str, title: str) -> str:
        """Create pie chart for proportions with truncation notice."""
        original_count = len(df)

        # Auto-select columns if not specified
        if not labels or not values:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            labels = categorical_cols[0] if categorical_cols else df.columns[0]
            values = numeric_cols[0] if numeric_cols else df.columns[1]

        # Limit to top 10 slices
        MAX_PIE_SLICES = 10
        if len(df) > MAX_PIE_SLICES:
            df = df.nlargest(MAX_PIE_SLICES, values)
            title = f"{title} (Top {MAX_PIE_SLICES} of {original_count} items)"

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Create pie chart
        wedges, texts, autotexts = ax.pie(
            df[values],
            labels=df[labels].astype(str),
            autopct='%1.1f%%',
            startangle=90,
            colors=self.colors['palette']
        )

        # Improve text readability
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')

        ax.axis('equal')

        return self._fig_to_base64(fig)

    def create_scatter_plot(self, df: pd.DataFrame, x: str, y: str, title: str) -> str:
        """Create scatter plot for correlations"""
        fig, ax = self._setup_chart(title, x or 'X', y or 'Y')

        # Auto-select columns if not specified
        if not x or not y:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x, y = numeric_cols[0], numeric_cols[1]
            else:
                x, y = df.columns[0], df.columns[1]

        ax.scatter(df[x], df[y], alpha=0.6, s=50, color=self.colors['primary'], edgecolors='white', linewidth=0.5)

        # Add trend line
        try:
            z = np.polyfit(df[x].dropna(), df[y].dropna(), 1)
            p = np.poly1d(z)
            ax.plot(df[x], p(df[x]), "r--", alpha=0.8, linewidth=2, label='Trend')
            ax.legend()
        except:
            pass  # Skip trend line if it fails

        ax.set_xlabel(x, fontsize=12)
        ax.set_ylabel(y, fontsize=12)

        return self._fig_to_base64(fig)

    def create_histogram(self, df: pd.DataFrame, column: str, bins: int, title: str) -> str:
        """Create distribution histogram"""
        fig, ax = self._setup_chart(title, column or 'Value', 'Frequency')

        # Auto-select column if not specified
        if not column:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            column = numeric_cols[0] if numeric_cols else df.columns[0]

        # Calculate optimal bins if not specified
        if bins is None or bins <= 0:
            bins = min(30, int(np.sqrt(len(df))))

        ax.hist(df[column].dropna(), bins=bins, color=self.colors['primary'], alpha=0.7, edgecolor='black')

        # Add statistical annotations
        mean_val = df[column].mean()
        median_val = df[column].median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()

        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)

        return self._fig_to_base64(fig)

    def create_heatmap(self, df: pd.DataFrame, title: str) -> str:
        """Create correlation heatmap"""
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

        # Select only numeric columns
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            raise ValueError("No numeric columns found for heatmap")

        # Calculate correlation matrix
        corr = numeric_df.corr()

        # Create heatmap
        sns.heatmap(
            corr,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)

        return self._fig_to_base64(fig)

    def create_box_plot(self, df: pd.DataFrame, columns: List[str], title: str) -> str:
        """Create box plot for statistical distribution"""
        fig, ax = self._setup_chart(title, 'Columns', 'Values')

        # Select numeric columns if not specified
        if not columns:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        if not columns:
            raise ValueError("No numeric columns found for box plot")

        # Limit to 10 columns for readability
        MAX_BOXPLOT_COLS = 10
        if len(columns) > MAX_BOXPLOT_COLS:
            columns = columns[:MAX_BOXPLOT_COLS]

        # Prepare data
        data_to_plot = [df[col].dropna() for col in columns]

        bp = ax.boxplot(data_to_plot, labels=columns, patch_artist=True)

        # Color the boxes
        for patch, color in zip(bp['boxes'], self.colors['palette']):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        plt.xticks(rotation=45, ha='right')
        ax.set_ylabel('Values', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        return self._fig_to_base64(fig)

    # Helper methods

    def _setup_chart(self, title: str, xlabel: str, ylabel: str) -> Tuple[plt.Figure, plt.Axes]:
        """Setup standard chart configuration"""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        return fig, ax

    def _fig_to_base64(self, fig: plt.Figure) -> str:
        """Convert matplotlib figure to base64 string"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64

    def _detect_date_columns(self, df: pd.DataFrame) -> List[str]:
        """Detect columns that contain date/time data"""
        date_cols = []

        # Check datetime columns
        date_cols.extend(df.select_dtypes(include=['datetime64']).columns.tolist())

        # Check object columns that might be dates
        for col in df.select_dtypes(include=['object']).columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    pd.to_datetime(df[col].head(10))
                    date_cols.append(col)
                except:
                    pass

        return date_cols

    def _has_percentage_data(self, df: pd.DataFrame, categorical_cols: List[str], numeric_cols: List[str]) -> bool:
        """Detect if data represents percentages or proportions"""
        if not categorical_cols or not numeric_cols:
            return False

        # Check if column names suggest percentages
        for col in numeric_cols:
            if 'percent' in col.lower() or 'proportion' in col.lower() or 'share' in col.lower():
                return True

        # Check if values sum to ~100 or ~1
        for col in numeric_cols:
            total = df[col].sum()
            if 95 <= total <= 105 or 0.95 <= total <= 1.05:
                return True

        # Check if data has few categories (good for pie chart)
        MAX_PIE_SLICES = 10
        if len(df) <= MAX_PIE_SLICES and df[categorical_cols[0]].nunique() <= MAX_PIE_SLICES:
            return True

        return False