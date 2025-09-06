import io
import json
import math
import time
from datetime import datetime, timedelta
from typing import List, Tuple, Dict, Any

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(
    page_title="Enhanced Process Mining Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        border: 1px solid #e6e9ef;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stAlert {
        margin-top: 1rem;
    }
    .process-insight {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 1rem;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.125rem;
    }
    .status-waiting { background-color: #fff3cd; color: #856404; }
    .status-approved { background-color: #d4edda; color: #155724; }
    .status-rejected { background-color: #f8d7da; color: #721c24; }
    .status-submitted { background-color: #d1ecf1; color: #0c5460; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Enhanced Utility Functions
# ----------------------------

@st.cache_data
def parse_datetime_safe(s):
    """Enhanced datetime parsing with multiple format support"""
    if pd.isna(s):
        return pd.NaT
    
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %H:%M",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(str(s), fmt)
        except Exception:
            pass
    
    try:
        return pd.to_datetime(s, errors="coerce")
    except Exception:
        return pd.NaT

@st.cache_data
def ensure_sorted_events(df, case_col, act_col, ts_col):
    """Sort events by case and timestamp"""
    d = df.copy()
    d[ts_col] = d[ts_col].apply(parse_datetime_safe)
    d = d.dropna(subset=[case_col, act_col, ts_col])
    d = d.sort_values([case_col, ts_col, act_col]).reset_index(drop=True)
    return d

@st.cache_data
def build_variants(df, case_col, act_col, ts_col):
    """Build process variants with enhanced analytics"""
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    seqs = d.groupby(case_col)[act_col].apply(list)
    variant_counts = seqs.value_counts().reset_index()
    variant_counts.columns = ["variant", "count"]
    variant_counts["variant_str"] = variant_counts["variant"].apply(lambda x: " → ".join(x))
    variant_counts["variant_length"] = variant_counts["variant"].apply(len)
    
    total_cases = variant_counts["count"].sum()
    variant_counts["percent"] = 100 * variant_counts["count"] / total_cases if total_cases else 0
    variant_counts["cumulative_percent"] = variant_counts["percent"].cumsum()
    
    return variant_counts, seqs

@st.cache_data
def directly_follows(df, case_col, act_col, ts_col):
    """Build directly-follows graph with transition probabilities"""
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    edges = {}
    activity_counts = {}
    
    for case, group in d.groupby(case_col):
        acts = group[act_col].tolist()
        for i, act in enumerate(acts):
            activity_counts[act] = activity_counts.get(act, 0) + 1
            if i < len(acts) - 1:
                edge = (acts[i], acts[i+1])
                edges[edge] = edges.get(edge, 0) + 1
    
    # Calculate transition probabilities
    edge_probs = {}
    for (source, target), count in edges.items():
        edge_probs[(source, target)] = count / activity_counts[source]
    
    return edges, edge_probs

@st.cache_data
def activity_durations(df, case_col, act_col, ts_col):
    """Compute enhanced activity duration statistics"""
    d = ensure_sorted_events(df, case_col, act_col, ts_col)
    records = []
    
    for case, g in d.groupby(case_col):
        ts = g[ts_col].tolist()
        acts = g[act_col].tolist()
        for i in range(len(acts)-1):
            delta = (ts[i+1] - ts[i]).total_seconds()
            if delta >= 0:
                records.append({
                    "case_id": case,
                    "activity": acts[i], 
                    "duration_s": delta,
                    "duration_h": delta / 3600.0,
                    "next_activity": acts[i+1]
                })
    
    if not records:
        return pd.DataFrame()
    
    df_dur = pd.DataFrame(records)
    
    # Enhanced aggregation
    agg = df_dur.groupby("activity")["duration_h"].agg([
        "count", "mean", "median", "std", "min", "max",
        lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
    ]).reset_index()
    
    agg.columns = ["activity", "count", "avg_hours", "median_hours", 
                   "std_hours", "min_hours", "max_hours", "q25_hours", "q75_hours"]
    
    return agg.sort_values("avg_hours", ascending=False)

@st.cache_data
def case_cycle_times(df, case_col, ts_col):
    """Enhanced case cycle time analysis"""
    d = ensure_sorted_events(df, case_col, ts_col, ts_col)
    agg = d.groupby(case_col)[ts_col].agg(["min", "max", "count"]).reset_index()
    agg["cycle_time_hours"] = (agg["max"] - agg["min"]).dt.total_seconds() / 3600.0
    agg["cycle_time_days"] = agg["cycle_time_hours"] / 24.0
    agg["event_count"] = agg["count"]
    
    # Add case start date for temporal analysis
    agg["start_date"] = agg["min"].dt.date
    agg["end_date"] = agg["max"].dt.date
    
    return agg

def edit_distance(a: List[str], b: List[str]) -> int:
    """Levenshtein distance with optimization"""
    if len(a) < len(b):
        a, b = b, a
    
    if len(b) == 0:
        return len(a)
    
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        current_row = [i + 1]
        for j, cb in enumerate(b):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (ca != cb)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

@st.cache_data
def conformance_analysis(seqs: pd.Series, ideal: List[str]) -> pd.DataFrame:
    """Enhanced conformance analysis with detailed metrics"""
    rows = []
    for case_id, seq in seqs.items():
        dist = edit_distance(seq, ideal)
        fitness = 1 - (dist / max(len(seq), len(ideal)))
        
        # Additional conformance metrics
        common_activities = set(seq) & set(ideal)
        precision = len(common_activities) / len(set(seq)) if seq else 0
        recall = len(common_activities) / len(set(ideal)) if ideal else 0
        
        rows.append({
            "case_id": case_id,
            "actual_sequence": " → ".join(seq),
            "edit_distance": dist,
            "fitness": fitness,
            "precision": precision,
            "recall": recall,
            "sequence_length": len(seq),
            "ideal_length": len(ideal)
        })
    
    return pd.DataFrame(rows)

def create_process_graph(edges, edge_probs, min_frequency=1):
    """Create enhanced process graph visualization"""
    G = nx.DiGraph()
    
    # Filter edges by frequency
    filtered_edges = {k: v for k, v in edges.items() if v >= min_frequency}
    
    for (source, target), weight in filtered_edges.items():
        prob = edge_probs.get((source, target), 0)
        G.add_edge(source, target, weight=weight, probability=prob)
    
    return G

def generate_insights(df, analysis_results):
    """Generate AI-powered insights"""
    insights = []
    
    # Variant insights
    if "variants" in analysis_results:
        variants = analysis_results["variants"]
        if len(variants) > 0:
            top_variant_pct = variants.iloc[0]["percent"]
            insights.append({
                "type": "variant",
                "title": "Process Standardization",
                "message": f"Your most common process variant accounts for {top_variant_pct:.1f}% of cases. Consider standardizing this as your 'happy path'.",
                "severity": "info" if top_variant_pct > 50 else "warning"
            })
    
    # Cycle time insights
    if "cycle_times" in analysis_results:
        cycle_times = analysis_results["cycle_times"]
        avg_days = cycle_times["cycle_time_days"].mean()
        std_days = cycle_times["cycle_time_days"].std()
        
        if std_days > avg_days:
            insights.append({
                "type": "performance",
                "title": "High Variability Detected",
                "message": f"Large variation in cycle times (avg: {avg_days:.1f} days, std: {std_days:.1f} days). Investigate process inconsistencies.",
                "severity": "warning"
            })
    
    # Activity duration insights
    if "activity_durations" in analysis_results:
        durations = analysis_results["activity_durations"]
        if not durations.empty:
            bottleneck = durations.iloc[0]
            insights.append({
                "type": "bottleneck",
                "title": "Potential Bottleneck",
                "message": f"'{bottleneck['activity']}' has the longest average duration ({bottleneck['avg_hours']:.1f} hours). Consider optimization.",
                "severity": "error" if bottleneck['avg_hours'] > 24 else "warning"
            })
    
    return insights

# ----------------------------
# Main Application
# ----------------------------

def main():
    st.title("🧭 Enhanced Process Mining Explorer")
    st.markdown("""
    **Discover, analyze, and optimize your business processes with advanced analytics**
    
    Upload your process data and gain insights into process variants, performance bottlenecks, 
    conformance issues, and optimization opportunities.
    """)
    
    # Sidebar configuration
    with st.sidebar:
        st.header("📁 Data Upload & Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload Process Data",
            type=["csv", "xlsx", "xls"],
            help="Upload your event log in CSV or Excel format"
        )
        
        if uploaded_file is not None:
            # Load data
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df)} events")
                
                # Column mapping
                st.subheader("🔧 Column Mapping")
                columns = df.columns.tolist()
                
                # Auto-detect columns
                case_col_default = next((col for col in columns if 'id' in col.lower()), columns[0])
                activity_col_default = next((col for col in columns if any(keyword in col.lower() 
                                           for keyword in ['status', 'activity', 'newvalue', 'state'])), columns[1] if len(columns) > 1 else columns[0])
                timestamp_col_default = next((col for col in columns if any(keyword in col.lower() 
                                            for keyword in ['time', 'date', 'created', 'timestamp'])), columns[-1])
                
                case_col = st.selectbox("Case ID Column", columns, index=columns.index(case_col_default))
                activity_col = st.selectbox("Activity/Status Column", columns, index=columns.index(activity_col_default))
                timestamp_col = st.selectbox("Timestamp Column", columns, index=columns.index(timestamp_col_default))
                
                # Optional columns
                resource_col = st.selectbox("Resource Column (Optional)", [""] + columns, index=0)
                attribute_cols = st.multiselect("Additional Attributes", 
                                              [col for col in columns if col not in [case_col, activity_col, timestamp_col]])
                
                # Analysis settings
                st.subheader("⚙️ Analysis Settings")
                min_variant_frequency = st.slider("Minimum Variant Frequency", 1, 50, 1)
                min_edge_frequency = st.slider("Minimum Edge Frequency (Process Map)", 1, 20, 1)
                
                # Process button
                if st.button("🔍 Analyze Process", type="primary"):
                    analyze_process(df, case_col, activity_col, timestamp_col, resource_col, 
                                  attribute_cols, min_variant_frequency, min_edge_frequency)
            
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
        
        else:
            st.info("👆 Upload a file to begin analysis")
            
            # Sample data info
            st.subheader("📋 Expected Data Format")
            st.markdown("""
            Your data should contain:
            - **Case ID**: Unique identifier for each case
            - **Activity/Status**: Process step or status
            - **Timestamp**: When the event occurred
            - **Resource** (optional): Who performed the activity
            
            Example for quote process:
            - Case ID: `QUO-12345-ABC`
            - Activity: `Waiting Submit`, `Approved`, etc.
            - Timestamp: `2024-08-06T07:36:40.000Z`
            """)

def analyze_process(df, case_col, activity_col, timestamp_col, resource_col, attribute_cols, 
                   min_variant_freq, min_edge_freq):
    """Main analysis function"""
    
    with st.spinner("🔄 Analyzing your process data..."):
        # Clean and prepare data
        df_clean = ensure_sorted_events(df, case_col, activity_col, timestamp_col)
        
        if df_clean.empty:
            st.error("No valid data found after cleaning. Please check your column mappings.")
            return
        
        # Store results
        analysis_results = {}
        
        # Basic metrics
        total_cases = df_clean[case_col].nunique()
        total_events = len(df_clean)
        unique_activities = df_clean[activity_col].nunique()
        date_range = (df_clean[timestamp_col].max() - df_clean[timestamp_col].min()).days
        
        # Calculate all analyses
        variants_df, sequences = build_variants(df_clean, case_col, activity_col, timestamp_col)
        edges, edge_probs = directly_follows(df_clean, case_col, activity_col, timestamp_col)
        activity_dur = activity_durations(df_clean, case_col, activity_col, timestamp_col)
        cycle_times = case_cycle_times(df_clean, case_col, timestamp_col)
        
        analysis_results = {
            "variants": variants_df,
            "sequences": sequences,
            "edges": edges,
            "edge_probs": edge_probs,
            "activity_durations": activity_dur,
            "cycle_times": cycle_times,
            "df_clean": df_clean
        }
    
    # Display results
    st.success("✅ Analysis complete!")
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🗺️ Process Map", "🧬 Variants", 
        "⏱️ Performance", "✅ Conformance", "💡 Insights"
    ])
    
    with tab1:
        show_overview(total_cases, total_events, unique_activities, date_range, 
                     cycle_times, variants_df, analysis_results)
    
    with tab2:
        show_process_map(edges, edge_probs, min_edge_freq, df_clean, activity_col)
    
    with tab3:
        show_variants_analysis(variants_df, sequences, min_variant_freq)
    
    with tab4:
        show_performance_analysis(cycle_times, activity_dur, df_clean, case_col, timestamp_col)
    
    with tab5:
        show_conformance_analysis(sequences, df_clean[activity_col].unique())
    
    with tab6:
        show_insights(df_clean, analysis_results)

def show_overview(total_cases, total_events, unique_activities, date_range, 
                 cycle_times, variants_df, analysis_results):
    """Display overview metrics and charts"""
    
    st.header("📊 Process Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", f"{total_cases:,}")
    with col2:
        st.metric("Total Events", f"{total_events:,}")
    with col3:
        st.metric("Unique Activities", unique_activities)
    with col4:
        st.metric("Date Range (days)", date_range)
    
    # Cycle time metrics
    if not cycle_times.empty:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Cycle Time", f"{cycle_times['cycle_time_hours'].mean():.1f} hours")
        with col2:
            st.metric("Median Cycle Time", f"{cycle_times['cycle_time_hours'].median():.1f} hours")
        with col3:
            st.metric("Min Cycle Time", f"{cycle_times['cycle_time_hours'].min():.1f} hours")
        with col4:
            st.metric("Max Cycle Time", f"{cycle_times['cycle_time_hours'].max():.1f} hours")
    
    # Timeline chart
    if not cycle_times.empty:
        st.subheader("📈 Case Timeline")
        
        # Daily case volume
        daily_cases = cycle_times.groupby('start_date').size().reset_index()
        daily_cases.columns = ['date', 'cases']
        
        fig = px.line(daily_cases, x='date', y='cases', 
                     title="Daily Case Volume",
                     labels={'cases': 'Number of Cases Started', 'date': 'Date'})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    # Process complexity overview
    col1, col2 = st.columns(2)
    
    with col1:
        if not variants_df.empty:
            # Variant distribution pie chart
            top_10_variants = variants_df.head(10)
            other_count = variants_df.iloc[10:]['count'].sum() if len(variants_df) > 10 else 0
            
            if other_count > 0:
                chart_data = pd.concat([
                    top_10_variants[['variant_str', 'count']],
                    pd.DataFrame([{'variant_str': 'Others', 'count': other_count}])
                ])
            else:
                chart_data = top_10_variants[['variant_str', 'count']]
            
            fig = px.pie(chart_data, values='count', names='variant_str',
                        title="Process Variant Distribution (Top 10)")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if not cycle_times.empty:
            # Cycle time distribution
            fig = px.histogram(cycle_times, x='cycle_time_hours', nbins=30,
                             title="Cycle Time Distribution",
                             labels={'cycle_time_hours': 'Cycle Time (hours)', 'count': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)

def show_process_map(edges, edge_probs, min_frequency, df_clean, activity_col):
    """Display process discovery map"""
    
    st.header("🗺️ Process Discovery Map")
    
    # Filter edges
    filtered_edges = {k: v for k, v in edges.items() if v >= min_frequency}
    
    if not filtered_edges:
        st.warning(f"No edges found with frequency >= {min_frequency}. Try lowering the threshold.")
        return
    
    # Create graph
    G = nx.DiGraph()
    for (source, target), weight in filtered_edges.items():
        prob = edge_probs.get((source, target), 0)
        G.add_edge(source, target, weight=weight, probability=prob)
    
    # Process map visualization using matplotlib
    fig, ax = plt.subplots(figsize=(15, 10))
    
    if len(G.nodes) > 1:
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 500 + 1000 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.8, ax=ax)
        
        # Draw edges with varying thickness
        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_weight = max(edge_weights) if edge_weights else 1
        edge_widths = [3 * (w / max_weight) + 0.5 for w in edge_weights]
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, 
                              edge_color='gray', arrows=True, 
                              arrowsize=20, arrowstyle='->', ax=ax)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
        
        # Add edge labels (frequencies)
        edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Process Flow Map", size=16, weight='bold')
        ax.axis('off')
    
    st.pyplot(fig)
    
    # Process statistics
    st.subheader("📈 Process Flow Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Most frequent transitions
        sorted_edges = sorted(filtered_edges.items(), key=lambda x: x[1], reverse=True)
        transition_df = pd.DataFrame([
            {"From": source, "To": target, "Frequency": freq, "Probability": f"{edge_probs.get((source, target), 0):.2%}"}
            for (source, target), freq in sorted_edges[:10]
        ])
        
        st.write("**Top Transitions**")
        st.dataframe(transition_df, use_container_width=True)
    
    with col2:
        # Activity statistics
        activity_stats = df_clean[activity_col].value_counts().head(10)
        
        fig = px.bar(x=activity_stats.values, y=activity_stats.index, 
                    orientation='h', title="Activity Frequency",
                    labels={'x': 'Frequency', 'y': 'Activity'})
        st.plotly_chart(fig, use_container_width=True)

def show_variants_analysis(variants_df, sequences, min_frequency):
    """Display process variants analysis"""
    
    st.header("🧬 Process Variants Analysis")
    
    if variants_df.empty:
        st.warning("No variants found in the data.")
        return
    
    # Filter by frequency
    filtered_variants = variants_df[variants_df['count'] >= min_frequency]
    
    # Variant metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Variants", len(variants_df))
    with col2:
        st.metric("Filtered Variants", len(filtered_variants))
    with col3:
        st.metric("Avg Variant Length", f"{variants_df['variant_length'].mean():.1f}")
    with col4:
        coverage = filtered_variants['percent'].sum()
        st.metric("Coverage", f"{coverage:.1f}%")
    
    # Pareto chart
    st.subheader("📊 Variant Pareto Analysis")
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart for frequencies
    fig.add_trace(
        go.Bar(x=list(range(1, len(filtered_variants) + 1)), 
               y=filtered_variants['count'],
               name="Frequency",
               marker_color='steelblue'),
        secondary_y=False,
    )
    
    # Line chart for cumulative percentage
    fig.add_trace(
        go.Scatter(x=list(range(1, len(filtered_variants) + 1)),
                  y=filtered_variants['cumulative_percent'],
                  mode='lines+markers',
                  name="Cumulative %",
                  line=dict(color='red', width=3)),
        secondary_y=True,
    )
    
    fig.update_xaxes(title_text="Variant Rank")
    fig.update_yaxes(title_text="Frequency", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
    fig.update_layout(title="Process Variant Pareto Chart")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed variant table
    st.subheader("📋 Variant Details")
    
    # Add selection for number of variants to display
    num_variants = st.selectbox("Number of variants to display:", [10, 25, 50, 100], index=1)
    
    display_variants = filtered_variants.head(num_variants)
    
    # Format the dataframe for display
    display_df = display_variants[['variant_str', 'count', 'percent', 'variant_length']].copy()
    display_df.columns = ['Process Variant', 'Cases', 'Percentage', 'Steps']
    display_df['Cases'] = display_df['Cases'].apply(lambda x: f"{x:,}")
    display_df['Percentage'] = display_df['Percentage'].apply(lambda x: f"{x:.1f}%")
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download button
    csv = variants_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Variants CSV",
        data=csv,
        file_name="process_variants.csv",
        mime="text/csv"
    )

def show_performance_analysis(cycle_times, activity_dur, df_clean, case_col, timestamp_col):
    """Display performance analysis"""
    
    st.header("⏱️ Performance Analysis")
    
    if cycle_times.empty:
        st.warning("No cycle time data available.")
        return
    
    # Performance metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Cycle Time Analysis")
        
        # Cycle time statistics
        stats_df = pd.DataFrame({
            'Metric': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', '95th Percentile'],
            'Hours': [
                cycle_times['cycle_time_hours'].mean(),
                cycle_times['cycle_time_hours'].median(),
                cycle_times['cycle_time_hours'].std(),
                cycle_times['cycle_time_hours'].min(),
                cycle_times['cycle_time_hours'].max(),
                cycle_times['cycle_time_hours'].quantile(0.95)
            ]
        })
        stats_df['Days'] = stats_df['Hours'] / 24
        stats_df['Hours'] = stats_df['Hours'].round(2)
        stats_df['Days'] = stats_df['Days'].round(2)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Cycle time box plot
        fig = px.box(cycle_times, y='cycle_time_hours',
                    title="Cycle Time Distribution (Box Plot)")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Activity Duration Analysis")
        
        if not activity_dur.empty:
            # Top 10 longest activities
            top_activities = activity_dur.head(10)
            
            fig = px.bar(top_activities, x='avg_hours', y='activity',
                        orientation='h', title="Average Activity Duration",
                        labels={'avg_hours': 'Average Hours', 'activity': 'Activity'})
            st.plotly_chart(fig, use_container_width=True)
            
            # Activity duration table
            display_dur = activity_dur[['activity', 'count', 'avg_hours', 'median_hours', 'std_hours']].copy()
            display_dur.columns = ['Activity', 'Count', 'Avg Hours', 'Median Hours', 'Std Hours']
            display_dur = display_dur.round(2)
            
            st.dataframe(display_dur, use_container_width=True)
    
    # Temporal analysis
    st.subheader("📅 Temporal Patterns")
    
    # Add day of week and hour analysis
    df_temp = df_clean.copy()
    df_temp['day_of_week'] = df_temp[timestamp_col].dt.day_name()
    df_temp['hour'] = df_temp[timestamp_col].dt.hour
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Events by day of week
        dow_counts = df_temp['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = dow_counts.reindex([day for day in dow_order if day in dow_counts.index])
        
        fig = px.bar(x=dow_counts.index, y=dow_counts.values,
                    title="Events by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Events by hour
        hour_counts = df_temp['hour'].value_counts().sort_index()
        
        fig = px.line(x=hour_counts.index, y=hour_counts.values,
                     title="Events by Hour of Day")
        st.plotly_chart(fig, use_container_width=True)

def show_conformance_analysis(sequences, unique_activities):
    """Display conformance analysis"""
    
    st.header("✅ Conformance Analysis")
    
    # Reference process input
    st.subheader("🎯 Define Reference Process")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Auto-suggest most common variant as reference
        if not sequences.empty:
            most_common = sequences.value_counts().index[0]
            default_reference = " → ".join(most_common)
        else:
            default_reference = ""
        
        reference_input = st.text_area(
            "Reference Process (comma or arrow separated):",
            value=default_reference,
            help="Enter the ideal process sequence, e.g., 'Start → Review → Approve → End'"
        )
    
    with col2:
        st.write("**Available Activities:**")
        for activity in sorted(unique_activities):
            st.write(f"• {activity}")
    
    if reference_input and not sequences.empty:
        # Parse reference process
        if "→" in reference_input:
            reference_process = [step.strip() for step in reference_input.split("→")]
        else:
            reference_process = [step.strip() for step in reference_input.split(",")]
        
        # Calculate conformance
        conformance_df = conformance_analysis(sequences, reference_process)
        
        if not conformance_df.empty:
            # Conformance metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_fitness = conformance_df['fitness'].mean()
                st.metric("Average Fitness", f"{avg_fitness:.2%}")
            
            with col2:
                perfect_cases = (conformance_df['fitness'] == 1).sum()
                st.metric("Perfect Cases", f"{perfect_cases} ({perfect_cases/len(conformance_df):.1%})")
            
            with col3:
                avg_precision = conformance_df['precision'].mean()
                st.metric("Average Precision", f"{avg_precision:.2%}")
            
            with col4:
                avg_recall = conformance_df['recall'].mean()
                st.metric("Average Recall", f"{avg_recall:.2%}")
            
            # Conformance distribution
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.histogram(conformance_df, x='fitness', nbins=20,
                                 title="Fitness Score Distribution",
                                 labels={'fitness': 'Fitness Score', 'count': 'Number of Cases'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.scatter(conformance_df, x='fitness', y='edit_distance',
                               color='sequence_length', size='sequence_length',
                               title="Fitness vs Edit Distance",
                               labels={'fitness': 'Fitness Score', 'edit_distance': 'Edit Distance'})
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed conformance table
            st.subheader("📋 Conformance Details")
            
            # Show worst conforming cases
            worst_cases = conformance_df.nsmallest(20, 'fitness')
            display_worst = worst_cases[['case_id', 'actual_sequence', 'fitness', 'edit_distance']].copy()
            display_worst['fitness'] = display_worst['fitness'].apply(lambda x: f"{x:.2%}")
            
            st.write("**Cases with Lowest Conformance:**")
            st.dataframe(display_worst, use_container_width=True)

def show_insights(df_clean, analysis_results):
    """Display AI-powered insights"""
    
    st.header("💡 AI-Powered Insights")
    
    # Generate insights
    insights = generate_insights(df_clean, analysis_results)
    
    if insights:
        for insight in insights:
            if insight['severity'] == 'error':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['severity'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.info(f"**{insight['title']}**: {insight['message']}")
    
    # Process optimization recommendations
    st.subheader("🚀 Optimization Recommendations")
    
    recommendations = []
    
    # Check for variants
    if "variants" in analysis_results and not analysis_results["variants"].empty:
        variants = analysis_results["variants"]
        if len(variants) > 10:
            recommendations.append({
                "category": "Process Standardization",
                "priority": "High",
                "recommendation": f"You have {len(variants)} process variants. Focus on standardizing the top 5 variants which likely cover 80% of your cases.",
                "impact": "Reduce process complexity and improve consistency"
            })
    
    # Check for bottlenecks
    if "activity_durations" in analysis_results and not analysis_results["activity_durations"].empty:
        durations = analysis_results["activity_durations"]
        if not durations.empty:
            longest_activity = durations.iloc[0]
            if longest_activity['avg_hours'] > 24:
                recommendations.append({
                    "category": "Bottleneck Removal",
                    "priority": "High",
                    "recommendation": f"Activity '{longest_activity['activity']}' takes {longest_activity['avg_hours']:.1f} hours on average. Investigate automation or process improvement opportunities.",
                    "impact": "Significantly reduce cycle time"
                })
    
    # Check for cycle time variability
    if "cycle_times" in analysis_results and not analysis_results["cycle_times"].empty:
        cycle_times = analysis_results["cycle_times"]
        cv = cycle_times['cycle_time_hours'].std() / cycle_times['cycle_time_hours'].mean()
        if cv > 1:
            recommendations.append({
                "category": "Process Standardization",
                "priority": "Medium",
                "recommendation": f"High variability in cycle times (CV: {cv:.1f}). Standardize process execution to reduce variation.",
                "impact": "Improve predictability and customer satisfaction"
            })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['category']} ({rec['priority']} Priority)"):
                st.write(f"**Recommendation:** {rec['recommendation']}")
                st.write(f"**Expected Impact:** {rec['impact']}")
    else:
        st.info("No specific recommendations available. Your process appears to be running efficiently!")
    
    # Export analysis results
    st.subheader("📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if "variants" in analysis_results:
            csv = analysis_results["variants"].to_csv(index=False)
            st.download_button("Download Variants", csv, "variants.csv", "text/csv")
    
    with col2:
        if "cycle_times" in analysis_results:
            csv = analysis_results["cycle_times"].to_csv(index=False)
            st.download_button("Download Cycle Times", csv, "cycle_times.csv", "text/csv")
    
    with col3:
        if "activity_durations" in analysis_results:
            csv = analysis_results["activity_durations"].to_csv(index=False)
            st.download_button("Download Activity Durations", csv, "activity_durations.csv", "text/csv")

# Run the app
if __name__ == "__main__":
    main()
