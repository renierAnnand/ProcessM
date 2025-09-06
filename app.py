import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from datetime import datetime, timedelta
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Process Mining Explorer",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
    .process-insight {
        background-color: #e8f4f8;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 0.5rem 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Utility Functions
# ----------------------------

def safe_parse_datetime(date_str):
    """Parse datetime with multiple format support"""
    if pd.isna(date_str) or date_str == '' or date_str is None:
        return pd.NaT
    
    # Convert to string if not already
    date_str = str(date_str).strip()
    
    # Common datetime formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ", 
        "%Y-%m-%dT%H:%M:%S",
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
            return datetime.strptime(date_str, fmt)
        except:
            continue
    
    # Try pandas parsing as last resort
    try:
        return pd.to_datetime(date_str, errors='coerce')
    except:
        return pd.NaT

def clean_and_sort_data(df, case_col, activity_col, timestamp_col):
    """Clean and sort the event data"""
    # Create a copy
    clean_df = df.copy()
    
    # Parse timestamps
    clean_df[timestamp_col] = clean_df[timestamp_col].apply(safe_parse_datetime)
    
    # Remove rows with missing essential data
    clean_df = clean_df.dropna(subset=[case_col, activity_col, timestamp_col])
    
    # Remove rows where timestamp parsing failed
    clean_df = clean_df[clean_df[timestamp_col].notna()]
    
    # Sort by case and timestamp
    clean_df = clean_df.sort_values([case_col, timestamp_col, activity_col]).reset_index(drop=True)
    
    return clean_df

def build_process_variants(df, case_col, activity_col, timestamp_col):
    """Build process variants"""
    # Group by case and create sequences
    sequences = df.groupby(case_col)[activity_col].apply(list).reset_index()
    sequences.columns = [case_col, 'sequence']
    
    # Convert sequences to strings for counting
    sequences['variant_str'] = sequences['sequence'].apply(lambda x: ' → '.join(x))
    sequences['variant_length'] = sequences['sequence'].apply(len)
    
    # Count variants
    variant_counts = sequences['variant_str'].value_counts().reset_index()
    variant_counts.columns = ['variant', 'count']
    variant_counts['variant_length'] = variant_counts['variant'].apply(lambda x: len(x.split(' → ')))
    
    total_cases = variant_counts['count'].sum()
    variant_counts['percent'] = (variant_counts['count'] / total_cases * 100).round(2)
    variant_counts['cumulative_percent'] = variant_counts['percent'].cumsum()
    
    return variant_counts, sequences

def calculate_directly_follows(df, case_col, activity_col, timestamp_col):
    """Calculate directly-follows relationships"""
    edges = {}
    activity_freq = {}
    
    # Group by case
    for case_id, group in df.groupby(case_col):
        activities = group.sort_values(timestamp_col)[activity_col].tolist()
        
        # Count activities
        for activity in activities:
            activity_freq[activity] = activity_freq.get(activity, 0) + 1
        
        # Count transitions
        for i in range(len(activities) - 1):
            edge = (activities[i], activities[i + 1])
            edges[edge] = edges.get(edge, 0) + 1
    
    # Calculate probabilities
    edge_probs = {}
    for (source, target), count in edges.items():
        if source in activity_freq and activity_freq[source] > 0:
            edge_probs[(source, target)] = count / activity_freq[source]
        else:
            edge_probs[(source, target)] = 0
    
    return edges, edge_probs, activity_freq

def calculate_cycle_times(df, case_col, timestamp_col):
    """Calculate case cycle times"""
    cycle_data = []
    
    for case_id, group in df.groupby(case_col):
        timestamps = group[timestamp_col].sort_values()
        if len(timestamps) >= 2:
            start_time = timestamps.iloc[0]
            end_time = timestamps.iloc[-1]
            cycle_time_hours = (end_time - start_time).total_seconds() / 3600
            
            cycle_data.append({
                case_col: case_id,
                'start_time': start_time,
                'end_time': end_time,
                'cycle_time_hours': cycle_time_hours,
                'cycle_time_days': cycle_time_hours / 24,
                'event_count': len(group),
                'start_date': start_time.date()
            })
    
    return pd.DataFrame(cycle_data)

def calculate_activity_durations(df, case_col, activity_col, timestamp_col):
    """Calculate activity durations"""
    duration_data = []
    
    for case_id, group in df.groupby(case_col):
        sorted_group = group.sort_values(timestamp_col)
        activities = sorted_group[activity_col].tolist()
        timestamps = sorted_group[timestamp_col].tolist()
        
        for i in range(len(activities) - 1):
            duration_hours = (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
            if duration_hours >= 0:
                duration_data.append({
                    'activity': activities[i],
                    'duration_hours': duration_hours,
                    'case_id': case_id
                })
    
    if not duration_data:
        return pd.DataFrame()
    
    df_durations = pd.DataFrame(duration_data)
    
    # Aggregate by activity
    activity_stats = df_durations.groupby('activity')['duration_hours'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).reset_index()
    
    activity_stats.columns = ['activity', 'count', 'avg_hours', 'median_hours', 'std_hours', 'min_hours', 'max_hours']
    activity_stats = activity_stats.fillna(0)
    
    return activity_stats.sort_values('avg_hours', ascending=False)

# ----------------------------
# Main Application
# ----------------------------

def main():
    st.title("🧭 Process Mining Explorer")
    st.markdown("""
    **Analyze and optimize your business processes**
    
    Upload your process event data to discover insights about process variants, 
    performance bottlenecks, and optimization opportunities.
    """)
    
    # Sidebar for file upload and configuration
    with st.sidebar:
        st.header("📁 Data Upload")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload your process event log"
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Loaded {len(df):,} rows")
                
                # Column mapping
                st.subheader("🔧 Column Mapping")
                columns = df.columns.tolist()
                
                # Auto-detect columns based on common patterns
                case_col_options = [col for col in columns if any(keyword in col.lower() for keyword in ['id', 'case', 'ticket', 'quo'])]
                activity_col_options = [col for col in columns if any(keyword in col.lower() for keyword in ['activity', 'status', 'state', 'newvalue', 'value'])]
                timestamp_col_options = [col for col in columns if any(keyword in col.lower() for keyword in ['time', 'date', 'created', 'timestamp', 'when'])]
                
                # Select columns
                case_col = st.selectbox(
                    "Case ID Column", 
                    columns, 
                    index=columns.index(case_col_options[0]) if case_col_options else 0
                )
                
                activity_col = st.selectbox(
                    "Activity/Status Column", 
                    columns,
                    index=columns.index(activity_col_options[0]) if activity_col_options else (1 if len(columns) > 1 else 0)
                )
                
                timestamp_col = st.selectbox(
                    "Timestamp Column", 
                    columns,
                    index=columns.index(timestamp_col_options[0]) if timestamp_col_options else (len(columns) - 1)
                )
                
                # Analysis settings
                st.subheader("⚙️ Settings")
                min_variant_freq = st.slider("Min Variant Frequency", 1, 10, 1)
                min_edge_freq = st.slider("Min Edge Frequency", 1, 5, 1)
                
                # Analyze button
                if st.button("🔍 Analyze Process", type="primary"):
                    analyze_process(df, case_col, activity_col, timestamp_col, min_variant_freq, min_edge_freq)
                    
            except Exception as e:
                st.error(f"Error loading file: {str(e)}")
                st.info("Please check your file format and try again.")
        
        else:
            st.info("👆 Upload a file to begin")
            
            # Show expected format
            st.subheader("📋 Expected Format")
            st.markdown("""
            Your data should have columns for:
            - **Case ID**: Unique identifier (e.g., QUO-12345)
            - **Activity**: Process step (e.g., "Waiting Submit", "Approved")
            - **Timestamp**: When event occurred
            """)

def analyze_process(df, case_col, activity_col, timestamp_col, min_variant_freq, min_edge_freq):
    """Main analysis function"""
    
    with st.spinner("🔄 Analyzing process data..."):
        # Clean and prepare data
        clean_df = clean_and_sort_data(df, case_col, activity_col, timestamp_col)
        
        if clean_df.empty:
            st.error("❌ No valid data found after cleaning. Please check your column mappings.")
            return
        
        # Calculate basic metrics
        total_cases = clean_df[case_col].nunique()
        total_events = len(clean_df)
        unique_activities = clean_df[activity_col].nunique()
        
        # Calculate analyses
        variants_df, sequences_df = build_process_variants(clean_df, case_col, activity_col, timestamp_col)
        edges, edge_probs, activity_freq = calculate_directly_follows(clean_df, case_col, activity_col, timestamp_col)
        cycle_times_df = calculate_cycle_times(clean_df, case_col, timestamp_col)
        activity_durations_df = calculate_activity_durations(clean_df, case_col, activity_col, timestamp_col)
        
        # Store results for tabs
        results = {
            'clean_df': clean_df,
            'variants_df': variants_df,
            'sequences_df': sequences_df,
            'edges': edges,
            'edge_probs': edge_probs,
            'activity_freq': activity_freq,
            'cycle_times_df': cycle_times_df,
            'activity_durations_df': activity_durations_df,
            'total_cases': total_cases,
            'total_events': total_events,
            'unique_activities': unique_activities
        }
    
    st.success("✅ Analysis complete!")
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview", 
        "🗺️ Process Map", 
        "🧬 Variants", 
        "⏱️ Performance", 
        "💡 Insights"
    ])
    
    with tab1:
        show_overview(results)
    
    with tab2:
        show_process_map(results, min_edge_freq)
    
    with tab3:
        show_variants(results, min_variant_freq)
    
    with tab4:
        show_performance(results)
    
    with tab5:
        show_insights(results)

def show_overview(results):
    """Show overview tab"""
    st.header("📊 Process Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Cases", f"{results['total_cases']:,}")
    with col2:
        st.metric("Total Events", f"{results['total_events']:,}")
    with col3:
        st.metric("Unique Activities", results['unique_activities'])
    with col4:
        if not results['cycle_times_df'].empty:
            avg_cycle = results['cycle_times_df']['cycle_time_hours'].mean()
            st.metric("Avg Cycle Time", f"{avg_cycle:.1f} hrs")
        else:
            st.metric("Avg Cycle Time", "N/A")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Top variants pie chart
        if not results['variants_df'].empty:
            top_variants = results['variants_df'].head(8)
            fig = px.pie(
                top_variants, 
                values='count', 
                names='variant',
                title="Top Process Variants"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Activity frequency
        if results['activity_freq']:
            activities = list(results['activity_freq'].keys())[:10]
            frequencies = [results['activity_freq'][act] for act in activities]
            
            fig = px.bar(
                x=frequencies,
                y=activities,
                orientation='h',
                title="Activity Frequency"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Timeline if we have cycle times
    if not results['cycle_times_df'].empty:
        st.subheader("📈 Timeline Analysis")
        daily_counts = results['cycle_times_df'].groupby('start_date').size().reset_index()
        daily_counts.columns = ['date', 'cases']
        
        fig = px.line(daily_counts, x='date', y='cases', title="Cases Started per Day")
        st.plotly_chart(fig, use_container_width=True)

def show_process_map(results, min_edge_freq):
    """Show process map tab"""
    st.header("🗺️ Process Discovery Map")
    
    # Filter edges by frequency
    filtered_edges = {k: v for k, v in results['edges'].items() if v >= min_edge_freq}
    
    if not filtered_edges:
        st.warning(f"No edges found with frequency >= {min_edge_freq}. Try lowering the threshold.")
        return
    
    # Create network graph
    G = nx.DiGraph()
    
    for (source, target), weight in filtered_edges.items():
        G.add_edge(source, target, weight=weight)
    
    # Create visualization
    if len(G.nodes) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        if len(G.nodes) > 1:
            pos = nx.spring_layout(G, k=2, iterations=50)
            
            # Draw nodes
            node_sizes = [results['activity_freq'].get(node, 100) * 50 + 500 for node in G.nodes()]
            nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.7, ax=ax)
            
            # Draw edges
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            max_weight = max(edge_weights) if edge_weights else 1
            edge_widths = [3 * (w / max_weight) + 0.5 for w in edge_weights]
            
            nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6, edge_color='gray', 
                                 arrows=True, arrowsize=20, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax)
            
            # Add edge labels
            edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, ax=ax)
        
        ax.set_title("Process Flow Map", size=14, weight='bold')
        ax.axis('off')
        st.pyplot(fig)
    
    # Show transition table
    st.subheader("📋 Top Transitions")
    
    if filtered_edges:
        transitions = []
        for (source, target), freq in sorted(filtered_edges.items(), key=lambda x: x[1], reverse=True)[:15]:
            prob = results['edge_probs'].get((source, target), 0)
            transitions.append({
                'From': source,
                'To': target,
                'Frequency': freq,
                'Probability': f"{prob:.1%}"
            })
        
        st.dataframe(pd.DataFrame(transitions), use_container_width=True)

def show_variants(results, min_variant_freq):
    """Show variants tab"""
    st.header("🧬 Process Variants")
    
    variants_df = results['variants_df']
    
    if variants_df.empty:
        st.warning("No variants found.")
        return
    
    # Filter variants
    filtered_variants = variants_df[variants_df['count'] >= min_variant_freq]
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Variants", len(variants_df))
    with col2:
        st.metric("Shown Variants", len(filtered_variants))
    with col3:
        st.metric("Coverage", f"{filtered_variants['percent'].sum():.1f}%")
    with col4:
        st.metric("Avg Length", f"{variants_df['variant_length'].mean():.1f}")
    
    # Pareto chart
    st.subheader("📊 Variant Distribution")
    
    top_variants = filtered_variants.head(20)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Bar chart
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(top_variants) + 1)),
            y=top_variants['count'],
            name="Cases",
            marker_color='steelblue'
        ),
        secondary_y=False
    )
    
    # Line chart for cumulative
    fig.add_trace(
        go.Scatter(
            x=list(range(1, len(top_variants) + 1)),
            y=top_variants['cumulative_percent'],
            mode='lines+markers',
            name="Cumulative %",
            line=dict(color='red', width=2)
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Variant Rank")
    fig.update_yaxes(title_text="Number of Cases", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative Percentage", secondary_y=True)
    fig.update_layout(title="Variant Pareto Chart")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Variants table
    st.subheader("📋 Variant Details")
    
    display_variants = filtered_variants.head(25).copy()
    display_variants['Cases'] = display_variants['count'].apply(lambda x: f"{x:,}")
    display_variants['Percentage'] = display_variants['percent'].apply(lambda x: f"{x:.1f}%")
    display_variants['Steps'] = display_variants['variant_length']
    
    st.dataframe(
        display_variants[['variant', 'Cases', 'Percentage', 'Steps']],
        use_container_width=True,
        column_config={
            'variant': 'Process Variant',
            'Cases': 'Cases',
            'Percentage': 'Percentage',
            'Steps': 'Steps'
        }
    )
    
    # Download
    csv = variants_df.to_csv(index=False)
    st.download_button(
        "📥 Download Variants CSV",
        csv,
        "process_variants.csv",
        "text/csv"
    )

def show_performance(results):
    """Show performance tab"""
    st.header("⏱️ Performance Analysis")
    
    cycle_times_df = results['cycle_times_df']
    activity_durations_df = results['activity_durations_df']
    
    if cycle_times_df.empty:
        st.warning("No cycle time data available.")
        return
    
    # Cycle time metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⏰ Cycle Time Statistics")
        
        stats = {
            'Mean': cycle_times_df['cycle_time_hours'].mean(),
            'Median': cycle_times_df['cycle_time_hours'].median(),
            'Std Dev': cycle_times_df['cycle_time_hours'].std(),
            'Min': cycle_times_df['cycle_time_hours'].min(),
            'Max': cycle_times_df['cycle_time_hours'].max(),
            '95th Percentile': cycle_times_df['cycle_time_hours'].quantile(0.95)
        }
        
        stats_df = pd.DataFrame([
            {'Metric': k, 'Hours': f"{v:.2f}", 'Days': f"{v/24:.2f}"}
            for k, v in stats.items()
        ])
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Cycle time distribution
        fig = px.histogram(
            cycle_times_df,
            x='cycle_time_hours',
            nbins=30,
            title="Cycle Time Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Activity Durations")
        
        if not activity_durations_df.empty:
            # Top 10 longest activities
            top_activities = activity_durations_df.head(10)
            
            fig = px.bar(
                top_activities,
                x='avg_hours',
                y='activity',
                orientation='h',
                title="Average Activity Duration"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Activity duration table
            display_dur = top_activities[['activity', 'count', 'avg_hours', 'median_hours']].copy()
            display_dur.columns = ['Activity', 'Count', 'Avg Hours', 'Median Hours']
            display_dur = display_dur.round(2)
            
            st.dataframe(display_dur, use_container_width=True)
        else:
            st.info("No activity duration data available.")
    
    # Temporal patterns
    st.subheader("📅 Temporal Patterns")
    
    clean_df = results['clean_df']
    timestamp_col = clean_df.columns[-1]  # Assuming last column is timestamp
    
    # Add time features
    temp_df = clean_df.copy()
    temp_df['hour'] = pd.to_datetime(temp_df[timestamp_col]).dt.hour
    temp_df['day_of_week'] = pd.to_datetime(temp_df[timestamp_col]).dt.day_name()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Events by hour
        hourly_counts = temp_df['hour'].value_counts().sort_index()
        fig = px.line(x=hourly_counts.index, y=hourly_counts.values, title="Events by Hour")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Events by day of week
        dow_counts = temp_df['day_of_week'].value_counts()
        dow_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_counts = dow_counts.reindex([day for day in dow_order if day in dow_counts.index])
        
        fig = px.bar(x=dow_counts.index, y=dow_counts.values, title="Events by Day of Week")
        st.plotly_chart(fig, use_container_width=True)

def show_insights(results):
    """Show insights tab"""
    st.header("💡 Process Insights")
    
    insights = []
    
    # Variant insights
    variants_df = results['variants_df']
    if not variants_df.empty:
        top_variant_pct = variants_df.iloc[0]['percent']
        num_variants = len(variants_df)
        
        if top_variant_pct < 50:
            insights.append({
                'type': 'warning',
                'title': 'High Process Variation',
                'message': f'Your most common variant only covers {top_variant_pct:.1f}% of cases. With {num_variants} total variants, consider process standardization.'
            })
        else:
            insights.append({
                'type': 'success',
                'title': 'Good Process Consistency',
                'message': f'Your most common variant covers {top_variant_pct:.1f}% of cases, indicating good process standardization.'
            })
    
    # Cycle time insights
    cycle_times_df = results['cycle_times_df']
    if not cycle_times_df.empty:
        avg_cycle = cycle_times_df['cycle_time_hours'].mean()
        std_cycle = cycle_times_df['cycle_time_hours'].std()
        cv = std_cycle / avg_cycle if avg_cycle > 0 else 0
        
        if cv > 1:
            insights.append({
                'type': 'warning',
                'title': 'High Cycle Time Variability',
                'message': f'Cycle time variation is high (CV: {cv:.1f}). This suggests inconsistent process execution.'
            })
        
        if avg_cycle > 168:  # More than a week
            insights.append({
                'type': 'error',
                'title': 'Long Cycle Times',
                'message': f'Average cycle time is {avg_cycle/24:.1f} days. Consider identifying bottlenecks.'
            })
    
    # Activity insights
    activity_durations_df = results['activity_durations_df']
    if not activity_durations_df.empty:
        longest_activity = activity_durations_df.iloc[0]
        if longest_activity['avg_hours'] > 24:
            insights.append({
                'type': 'warning',
                'title': 'Potential Bottleneck',
                'message': f"Activity '{longest_activity['activity']}' takes {longest_activity['avg_hours']:.1f} hours on average. This may be a bottleneck."
            })
    
    # Display insights
    if insights:
        for insight in insights:
            if insight['type'] == 'error':
                st.error(f"**{insight['title']}**: {insight['message']}")
            elif insight['type'] == 'warning':
                st.warning(f"**{insight['title']}**: {insight['message']}")
            else:
                st.success(f"**{insight['title']}**: {insight['message']}")
    else:
        st.info("No specific insights generated. Your process appears to be running smoothly!")
    
    # Recommendations
    st.subheader("🚀 Recommendations")
    
    recommendations = []
    
    if not variants_df.empty and len(variants_df) > 10:
        recommendations.append({
            'title': 'Process Standardization',
            'description': f'You have {len(variants_df)} process variants. Focus on standardizing the top 5 variants.',
            'impact': 'High'
        })
    
    if not activity_durations_df.empty:
        long_activities = activity_durations_df[activity_durations_df['avg_hours'] > 8]
        if not long_activities.empty:
            recommendations.append({
                'title': 'Optimize Long-Running Activities',
                'description': f'{len(long_activities)} activities take more than 8 hours on average. Consider automation or process improvement.',
                'impact': 'Medium'
            })
    
    if not cycle_times_df.empty:
        cv = cycle_times_df['cycle_time_hours'].std() / cycle_times_df['cycle_time_hours'].mean()
        if cv > 0.5:
            recommendations.append({
                'title': 'Reduce Process Variability',
                'description': 'High variation in cycle times suggests inconsistent execution. Implement standard operating procedures.',
                'impact': 'Medium'
            })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            with st.expander(f"{i}. {rec['title']} (Impact: {rec['impact']})"):
                st.write(rec['description'])
    else:
        st.success("No specific recommendations. Your process is performing well!")
    
    # Export options
    st.subheader("📥 Export Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if not variants_df.empty:
            csv = variants_df.to_csv(index=False)
            st.download_button("Variants", csv, "variants.csv", "text/csv")
    
    with col2:
        if not cycle_times_df.empty:
            csv = cycle_times_df.to_csv(index=False)
            st.download_button("Cycle Times", csv, "cycle_times.csv", "text/csv")
    
    with col3:
        if not activity_durations_df.empty:
            csv = activity_durations_df.to_csv(index=False)
            st.download_button("Activity Durations", csv, "activity_durations.csv", "text/csv")

# Run the app
if __name__ == "__main__":
    main()
