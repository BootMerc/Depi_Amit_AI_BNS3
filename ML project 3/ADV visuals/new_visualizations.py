# NEW VISUALIZATIONS FOR CUSTOMER SEGMENTATION
# Add these cells to your customer_segments.ipynb

# ============================================
# CELL 1: Import Additional Libraries
# ============================================
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler

# ============================================
# CELL 2: Parallel Coordinates Plot
# ============================================
def create_parallel_coordinates(data_with_clusters):
    """
    Parallel coordinates visualization for multi-dimensional data
    Shows how each customer's spending patterns compare across all features
    """
    # Convert from log scale back to original
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster']

    # Normalize for better visualization
    scaler = MinMaxScaler()
    normalized = pd.DataFrame(
        scaler.fit_transform(original_data.drop('Cluster', axis=1)),
        columns=original_data.drop('Cluster', axis=1).columns
    )
    normalized['Cluster'] = original_data['Cluster']

    fig = go.Figure(data=
        go.Parcoords(
            line = dict(
                color = normalized['Cluster'],
                colorscale = 'Viridis',
                showscale = True,
                cmin = normalized['Cluster'].min(),
                cmax = normalized['Cluster'].max()
            ),
            dimensions = list([
                dict(range = [0,1],
                     label = 'Fresh', 
                     values = normalized['Fresh']),
                dict(range = [0,1],
                     label = 'Milk', 
                     values = normalized['Milk']),
                dict(range = [0,1],
                     label = 'Grocery', 
                     values = normalized['Grocery']),
                dict(range = [0,1],
                     label = 'Frozen', 
                     values = normalized['Frozen']),
                dict(range = [0,1],
                     label = 'Detergents_Paper', 
                     values = normalized['Detergents_Paper']),
                dict(range = [0,1],
                     label = 'Delicatessen', 
                     values = normalized['Delicatessen'])
            ])
        )
    )

    fig.update_layout(
        title='Parallel Coordinates Plot - Customer Spending Patterns',
        height=600,
        font=dict(size=12)
    )

    return fig

# Generate the plot (assuming you have 'good_data' with 'Cluster' column)
# Replace 'good_data' with your clustered data variable name
fig_parcoords = create_parallel_coordinates(good_data)
fig_parcoords.show()

# ============================================
# CELL 3: 3D Scatter Plot
# ============================================
def create_3d_scatter(pca_results, data_with_clusters, z_feature='Fresh'):
    """
    3D scatter plot using PCA components + one original feature
    Helps visualize cluster separation in 3D space
    """
    plot_data = pca_results.copy()
    plot_data['Z_axis'] = data_with_clusters[z_feature]
    plot_data['Cluster'] = data_with_clusters['Cluster'].astype(str)

    fig = px.scatter_3d(
        plot_data,
        x='Dimension 1',  # or 'PC1' depending on your column names
        y='Dimension 2',  # or 'PC2'
        z='Z_axis',
        color='Cluster',
        title=f'3D Scatter Plot - PCA Components + {z_feature} Spending (log scale)',
        labels={
            'Dimension 1': 'Principal Component 1',
            'Dimension 2': 'Principal Component 2',
            'Z_axis': f'{z_feature} (log scale)'
        },
        color_discrete_sequence=px.colors.qualitative.Set2,
        height=700
    )

    fig.update_traces(marker=dict(size=5, opacity=0.8))

    return fig

# Generate the plot
# Replace variable names with your actual PCA results and clustered data
fig_3d = create_3d_scatter(pca_results, good_data, z_feature='Fresh')
fig_3d.show()

# You can try different features for Z-axis:
# fig_3d_milk = create_3d_scatter(pca_results, good_data, z_feature='Milk')
# fig_3d_milk.show()

# ============================================
# CELL 4: Sunburst Chart
# ============================================
def create_sunburst_chart(data_with_clusters):
    """
    Sunburst chart showing hierarchical customer segmentation
    Hierarchy: All Customers -> Cluster -> Fresh Category -> Milk Category
    """
    # Convert to original scale
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster'].astype(str)

    # Categorize spending levels
    def categorize_spending(series):
        q33 = series.quantile(0.33)
        q66 = series.quantile(0.66)
        return pd.cut(
            series, 
            bins=[0, q33, q66, float('inf')], 
            labels=['Low', 'Medium', 'High']
        )

    original_data['Fresh_Cat'] = categorize_spending(original_data['Fresh'])
    original_data['Milk_Cat'] = categorize_spending(original_data['Milk'])
    original_data['Grocery_Cat'] = categorize_spending(original_data['Grocery'])

    # Build hierarchical structure
    labels = []
    parents = []
    values = []

    # Root
    labels.append("All Customers")
    parents.append("")
    values.append(len(original_data))

    # Level 1: Clusters
    for cluster in original_data['Cluster'].unique():
        cluster_count = (original_data['Cluster'] == cluster).sum()
        labels.append(f"Cluster {cluster}")
        parents.append("All Customers")
        values.append(cluster_count)

    # Level 2: Fresh categories
    for cluster in original_data['Cluster'].unique():
        for fresh_cat in ['Low', 'Medium', 'High']:
            mask = (original_data['Cluster'] == cluster) & (original_data['Fresh_Cat'] == fresh_cat)
            count = mask.sum()
            if count > 0:
                labels.append(f"{fresh_cat} Fresh")
                parents.append(f"Cluster {cluster}")
                values.append(count)

    # Level 3: Milk categories (only significant groups)
    for cluster in original_data['Cluster'].unique():
        for fresh_cat in ['Low', 'Medium', 'High']:
            for milk_cat in ['Low', 'Medium', 'High']:
                mask = (
                    (original_data['Cluster'] == cluster) & 
                    (original_data['Fresh_Cat'] == fresh_cat) & 
                    (original_data['Milk_Cat'] == milk_cat)
                )
                count = mask.sum()
                if count > 5:  # Filter small groups
                    labels.append(f"{milk_cat} Milk")
                    parents.append(f"{fresh_cat} Fresh")
                    values.append(count)

    fig = go.Figure(go.Sunburst(
        labels=labels,
        parents=parents,
        values=values,
        branchvalues="total",
        marker=dict(colorscale='RdBu', showscale=True),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>%{percentParent}<extra></extra>'
    ))

    fig.update_layout(
        title='Sunburst Chart - Hierarchical Customer Segmentation',
        height=700
    )

    return fig

# Generate the plot
fig_sunburst = create_sunburst_chart(good_data)
fig_sunburst.show()

# ============================================
# CELL 5: Bonus - Interactive Cluster Comparison
# ============================================
def create_cluster_comparison_heatmap(data_with_clusters):
    """
    Heatmap comparing average spending across clusters
    """
    original_data = np.exp(data_with_clusters.drop('Cluster', axis=1))
    original_data['Cluster'] = data_with_clusters['Cluster']

    cluster_means = original_data.groupby('Cluster').mean()

    fig = px.imshow(
        cluster_means,
        labels=dict(x="Product Category", y="Cluster", color="Avg Spending"),
        x=cluster_means.columns,
        y=[f"Cluster {i}" for i in cluster_means.index],
        color_continuous_scale="Blues",
        aspect="auto",
        title="Heatmap - Average Spending by Cluster"
    )

    fig.update_layout(height=400)

    return fig

fig_heatmap = create_cluster_comparison_heatmap(good_data)
fig_heatmap.show()

print("✅ All new visualizations created!")
print("\nVisualization Summary:")
print("1. Parallel Coordinates - Multi-dimensional comparison")
print("2. 3D Scatter - Spatial cluster separation")
print("3. Sunburst Chart - Hierarchical segmentation")
print("4. Heatmap - Cluster comparison")
