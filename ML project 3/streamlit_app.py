import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from collections import Counter

# Page config
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    /* Fix Streamlit metric colors for dark theme */
    div[data-testid="stMetric"] {
        background-color: #20232a !important;  /* Much darker than default */
        color: #FFF !important;
        border-radius: 12px;
        padding: 20px;
        font-size: 2rem;
        box-shadow: 0 1px 4px #0001;
    }
    /* Make metric numbers pure white, bold */
    div[data-testid="stMetric"] > div:first-child {
        color: #FFF !important;
        font-weight: bold;
        font-size: 2.6rem;
        letter-spacing: 1px;
    }
    /* Label color slightly grayed for readability */
    div[data-testid="stMetric"] > div:last-child {
        color: #D3D3D3 !important;
        font-size: 1.1rem;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_process_data():
    """Load data and perform clustering"""
    try:
        data = pd.read_csv('customers.csv')
        data = data.drop(['Region', 'Channel'], axis=1, errors='ignore')
    except:
        # Fallback: generate sample data if file not found
        np.random.seed(42)
        data = pd.DataFrame({
            'Fresh': np.random.lognormal(9, 1.2, 440),
            'Milk': np.random.lognormal(8.5, 1.1, 440),
            'Grocery': np.random.lognormal(8.8, 1.0, 440),
            'Frozen': np.random.lognormal(7.5, 1.3, 440),
            'Detergents_Paper': np.random.lognormal(7.8, 1.4, 440),
            'Delicatessen': np.random.lognormal(7.2, 1.5, 440)
        })

    # Log transform
    log_data = np.log(data)

    # Outlier removal
    outliers = []
    for feature in log_data.columns:
        Q1 = np.percentile(log_data[feature], 25)
        Q3 = np.percentile(log_data[feature], 75)
        step = 1.5 * (Q3 - Q1)
        feature_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))].index
        outliers.extend(feature_outliers)

    outliers = Counter(outliers)
    multiple_outliers = [k for k, v in outliers.items() if v >= 2]
    good_data = log_data.drop(log_data.index[multiple_outliers]).reset_index(drop=True)

    return data, good_data

@st.cache_data
def perform_clustering(good_data, n_clusters):
    """Perform PCA and clustering"""
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(good_data)

    clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = clusterer.fit_predict(reduced_data)

    pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = clusters

    result_df = good_data.copy()
    result_df['Cluster'] = clusters

    return pca_df, result_df, pca, clusterer


def create_parcoords(data_with_clusters):
    """Parallel coordinates plot"""
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster']

    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(original_data.drop('Cluster', axis=1)),
        columns=original_data.drop('Cluster', axis=1).columns
    )
    normalized['Cluster'] = original_data['Cluster']

    fig = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=normalized['Cluster'],
                colorscale='Viridis',
                showscale=True,
                cmin=normalized['Cluster'].min(),
                cmax=normalized['Cluster'].max()
            ),
            dimensions=[
                dict(range=[0,1], label=col, values=normalized[col])
                for col in normalized.columns if col != 'Cluster'
            ]
        )
    )
    fig.update_layout(
        title='Parallel Coordinates - Feature Comparison Across Clusters',
        height=500,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    return fig

def create_3d_scatter(pca_df, data_with_clusters):
    """3D scatter plot"""
    plot_data = pca_df.copy()
    plot_data['Fresh'] = data_with_clusters['Fresh']
    plot_data['Cluster'] = plot_data['Cluster'].astype(str)

    fig = px.scatter_3d(
        plot_data,
        x='PC1', y='PC2', z='Fresh',
        color='Cluster',
        title='3D Visualization - PCA Components + Fresh Spending',
        labels={'PC1': 'PC 1', 'PC2': 'PC 2', 'Fresh': 'Fresh (log)'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=600
    )
    fig.update_traces(marker=dict(size=4, opacity=0.7))
    return fig

def create_sunburst(data_with_clusters):
    """Sunburst hierarchical chart"""
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster'].astype(str)

    def categorize(series):
        q33, q66 = series.quantile([0.33, 0.66])
        return pd.cut(series, bins=[0, q33, q66, float('inf')], labels=['Low', 'Med', 'High'])

    original_data['Fresh_Cat'] = categorize(original_data['Fresh'])
    original_data['Milk_Cat'] = categorize(original_data['Milk'])

    labels, parents, values = ["All"], [""], [len(original_data)]

    # Cluster level
    for cluster in original_data['Cluster'].unique():
        count = (original_data['Cluster'] == cluster).sum()
        labels.append(f"Cluster {cluster}")
        parents.append("All")
        values.append(count)

    # Fresh category level
    for cluster in original_data['Cluster'].unique():
        for cat in ['Low', 'Med', 'High']:
            count = ((original_data['Cluster'] == cluster) & (original_data['Fresh_Cat'] == cat)).sum()
            if count > 0:
                labels.append(f"{cat} Fresh")
                parents.append(f"Cluster {cluster}")
                values.append(count)

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colorscale='Blues'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    ))
    fig.update_layout(
        title='Sunburst - Hierarchical Segmentation',
        height=600
    )
    return fig

def create_cluster_comparison(data_with_clusters):
    """Cluster comparison bar chart"""
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster']

    cluster_means = original_data.groupby('Cluster').mean()

    fig = go.Figure()
    for col in cluster_means.columns:
        fig.add_trace(go.Bar(
            name=col,
            x=cluster_means.index.astype(str),
            y=cluster_means[col],
            text=cluster_means[col].round(0),
            textposition='outside'
        ))

    fig.update_layout(
        title='Average Spending by Cluster',
        xaxis_title='Cluster',
        yaxis_title='Average Spending (monetary units)',
        barmode='group',
        height=400
    )
    return fig

def create_pca_scatter(pca_df):
    """2D PCA scatter"""
    plot_data = pca_df.copy()
    plot_data['Cluster'] = plot_data['Cluster'].astype(str)

    fig = px.scatter(
        plot_data,
        x='PC1', y='PC2',
        color='Cluster',
        title='PCA Projection - Customer Segments',
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=400
    )
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    return fig


def main():
    st.title("📊 Customer Segmentation Analytics")
    st.markdown("### Interactive ML Dashboard for Wholesale Customer Analysis")

    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 5, 2)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 About")
    st.sidebar.info(
        "This dashboard analyzes wholesale customer data using "
        "unsupervised learning (PCA + K-Means). Explore spending patterns "
        "across 6 product categories."
    )

    # Load data
    with st.spinner("Loading data..."):
        raw_data, processed_data = load_and_process_data()
        pca_df, clustered_data, pca_model, kmeans_model = perform_clustering(processed_data, n_clusters)

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Customers", len(raw_data))
    with col2:
        st.metric("Features", len(raw_data.columns))
    with col3:
        st.metric("Clusters", n_clusters)
    with col4:
        st.metric("Outliers Removed", len(raw_data) - len(processed_data))

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Advanced Viz", "📊 Cluster Analysis", "📋 Data"])

    with tab1:
        st.subheader("Key Visualizations")

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(create_pca_scatter(pca_df), use_container_width=True)
        with col2:
            st.plotly_chart(create_cluster_comparison(clustered_data), use_container_width=True)

        st.markdown('<div class="insight-box"><b>💡 Insight:</b> The PCA projection shows clear separation between customer segments, indicating distinct purchasing behaviors.</div>', unsafe_allow_html=True)

    with tab2:
        st.subheader("Advanced Visualizations")

        viz_option = st.radio(
            "Select Visualization",
            ["Parallel Coordinates", "3D Scatter", "Sunburst Chart"],
            horizontal=True
        )

        if viz_option == "Parallel Coordinates":
            st.plotly_chart(create_parcoords(clustered_data), use_container_width=True)
            st.markdown("**How to use:** Drag lines to filter. Each line represents a customer across all product categories.")

        elif viz_option == "3D Scatter":
            st.plotly_chart(create_3d_scatter(pca_df, clustered_data), use_container_width=True)
            st.markdown("**How to use:** Rotate and zoom to explore 3D relationships. Third axis shows Fresh product spending.")

        else:
            st.plotly_chart(create_sunburst(clustered_data), use_container_width=True)
            st.markdown("**How to use:** Click segments to zoom in. Hierarchy: All → Cluster → Fresh Category → Milk Category")

    with tab3:
        st.subheader("Cluster Deep Dive")

        selected_cluster = st.selectbox("Select Cluster", sorted(clustered_data['Cluster'].unique()))

        cluster_data = clustered_data[clustered_data['Cluster'] == selected_cluster]
        original_cluster = np.exp(cluster_data.drop('Cluster', axis=1))

        st.markdown(f"### Cluster {selected_cluster} Statistics")
        st.markdown(f"**Size:** {len(cluster_data)} customers ({len(cluster_data)/len(clustered_data)*100:.1f}%)")

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Mean Spending**")
            st.dataframe(original_cluster.mean().round(0).to_frame('Mean').T, use_container_width=True)
        with col2:
            st.write("**Std Deviation**")
            st.dataframe(original_cluster.std().round(0).to_frame('Std').T, use_container_width=True)

        # Feature importance for cluster
        st.markdown("### Feature Distribution")
        feature = st.selectbox("Select Feature", original_cluster.columns)

        fig = px.histogram(
            original_cluster,
            x=feature,
            nbins=30,
            title=f'{feature} Distribution in Cluster {selected_cluster}'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Raw Data Preview")

        show_original = st.checkbox("Show original scale", value=True)

        if show_original:
            display_data = np.exp(clustered_data.drop('Cluster', axis=1))
            display_data['Cluster'] = clustered_data['Cluster']
        else:
            display_data = clustered_data

        st.dataframe(display_data.head(100), use_container_width=True)

        # Download
        csv = display_data.to_csv(index=False)
        st.download_button(
            "📥 Download Full Dataset",
            csv,
            "customer_segments.csv",
            "text/csv"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit + Plotly | Data Science Dashboard"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
