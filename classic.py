
from typing import Tuple, Optional, List
import numpy as np
from osgeo import gdal
from scipy import ndimage
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class BAGAnomalyDetector:
    """
    A class to detect anomalies in bathymetric BAG files using various statistical and ML methods.
    """

    def __init__(self, bag_file_path: str):
        """Initialize the BAG anomaly detector."""
        self.bag_file_path = bag_file_path
        self.dataset = None
        self.elevation_band = None
        self.uncertainty_band = None
        self.elevation_data = None
        self.uncertainty_data = None
        self.nodata_value = None

    def load_bag_file(self) -> bool:
        """Load the BAG file and extract elevation and uncertainty bands."""
        try:
            self.dataset = gdal.Open(self.bag_file_path, gdal.GA_ReadOnly)
            if self.dataset is None:
                raise ValueError(f"Could not open BAG file: {self.bag_file_path}")

            print(f"Successfully opened BAG file: {self.bag_file_path}")
            print(f"Raster size: {self.dataset.RasterXSize} x {self.dataset.RasterYSize}")
            print(f"Number of bands: {self.dataset.RasterCount}")

            self.elevation_band = self.dataset.GetRasterBand(1)
            self.nodata_value = self.elevation_band.GetNoDataValue()

            elevation_raw = self.elevation_band.ReadAsArray().astype(np.float64)

            if self.nodata_value is not None:
                self.elevation_data = np.where(elevation_raw == self.nodata_value, np.nan, elevation_raw)
            else:
                self.elevation_data = elevation_raw

            print(f"Elevation data range: {np.nanmin(self.elevation_data):.2f} to {np.nanmax(self.elevation_data):.2f}")

            # Get uncertainty band (band 2) if available
            if self.dataset.RasterCount >= 2:
                self.uncertainty_band = self.dataset.GetRasterBand(2)
                uncertainty_raw = self.uncertainty_band.ReadAsArray().astype(np.float64)

                uncertainty_nodata = self.uncertainty_band.GetNoDataValue()
                if uncertainty_nodata is not None:
                    self.uncertainty_data = np.where(uncertainty_raw == uncertainty_nodata, np.nan, uncertainty_raw)
                else:
                    self.uncertainty_data = uncertainty_raw

                print(f"Uncertainty data range: {np.nanmin(self.uncertainty_data):.2f} to {np.nanmax(self.uncertainty_data):.2f}")
            else:
                print("No uncertainty band found (band 2)")

            return True

        except Exception as e:
            print(f"Error loading BAG file: {str(e)}")
            return False

    def detect_statistical_anomalies(self, z_threshold: float = 3.0) -> np.ndarray:
        """Detect anomalies using Z-score method."""
        if self.elevation_data is None:
            raise ValueError("Elevation data not loaded")

        valid_mask = ~np.isnan(self.elevation_data)
        if not np.any(valid_mask):
            return np.zeros_like(self.elevation_data, dtype=bool)

        valid_data = self.elevation_data[valid_mask]
        mean_val = np.mean(valid_data)
        std_val = np.std(valid_data)

        if std_val == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        z_scores = np.abs((self.elevation_data - mean_val) / std_val)
        anomalies = (z_scores > z_threshold) & valid_mask

        print(f"Statistical anomalies detected: {np.sum(anomalies)} points ({np.sum(anomalies)/anomalies.size*100:.2f}%)")
        return anomalies

    def detect_gradient_anomalies(self, gradient_threshold: float = None) -> np.ndarray:
        """Detect anomalies based on steep gradients."""
        if self.elevation_data is None:
            raise ValueError("Elevation data not loaded")

        valid_mask = ~np.isnan(self.elevation_data)
        if not np.any(valid_mask):
            return np.zeros_like(self.elevation_data, dtype=bool)

        filled_data = np.copy(self.elevation_data)
        filled_data[~valid_mask] = np.nanmean(self.elevation_data)

        gradient_x = np.gradient(filled_data, axis=1)
        gradient_y = np.gradient(filled_data, axis=0)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

        gradient_magnitude[~valid_mask] = np.nan

        valid_gradients = gradient_magnitude[~np.isnan(gradient_magnitude)]
        if len(valid_gradients) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        if gradient_threshold is None:
            gradient_threshold = np.percentile(valid_gradients, 95)

        anomalies = (gradient_magnitude > gradient_threshold) & valid_mask

        print(f"Gradient anomalies detected: {np.sum(anomalies)} points ({np.sum(anomalies)/anomalies.size*100:.2f}%)")
        return anomalies

    def detect_morphological_anomalies(self, structure_size: int = 3) -> np.ndarray:
        """Detect anomalies using morphological operations."""
        if self.elevation_data is None:
            raise ValueError("Elevation data not loaded")

        valid_mask = ~np.isnan(self.elevation_data)
        if not np.any(valid_mask):
            return np.zeros_like(self.elevation_data, dtype=bool)

        filled_data = np.copy(self.elevation_data)
        filled_data[~valid_mask] = np.nanmean(self.elevation_data)

        structure = np.ones((structure_size, structure_size))

        opened = ndimage.grey_opening(filled_data, structure=structure)
        closed = ndimage.grey_closing(filled_data, structure=structure)

        spike_diff = filled_data - opened
        hole_diff = closed - filled_data

        spike_diff[~valid_mask] = 0
        hole_diff[~valid_mask] = 0

        valid_spike_diff = spike_diff[valid_mask]
        valid_hole_diff = hole_diff[valid_mask]

        if len(valid_spike_diff) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        spike_threshold = np.percentile(valid_spike_diff, 95) if len(valid_spike_diff) > 0 else 0
        hole_threshold = np.percentile(valid_hole_diff, 95) if len(valid_hole_diff) > 0 else 0

        spikes = (spike_diff > spike_threshold) & valid_mask
        holes = (hole_diff > hole_threshold) & valid_mask
        anomalies = spikes | holes

        print(f"Morphological anomalies detected: {np.sum(anomalies)} points ({np.sum(anomalies)/anomalies.size*100:.2f}%)")
        return anomalies

    def detect_uncertainty_anomalies(self, uncertainty_threshold: float = None) -> Optional[np.ndarray]:
        """Detect anomalies based on high uncertainty values."""
        if self.uncertainty_data is None:
            print("No uncertainty data available for anomaly detection")
            return None

        valid_mask = ~np.isnan(self.uncertainty_data)
        if not np.any(valid_mask):
            return np.zeros_like(self.uncertainty_data, dtype=bool)

        if uncertainty_threshold is None:
            valid_uncertainty = self.uncertainty_data[valid_mask]
            uncertainty_threshold = np.percentile(valid_uncertainty, 90)

        anomalies = (self.uncertainty_data > uncertainty_threshold) & valid_mask

        print(f"Uncertainty anomalies detected: {np.sum(anomalies)} points ({np.sum(anomalies)/anomalies.size*100:.2f}%)")
        return anomalies

    def extract_features(self, window_size: int = 5) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Extract features for ML methods."""
        if self.elevation_data is None:
            raise ValueError("Elevation data not loaded")

        print(f"Extracting features with window size {window_size}...")

        rows, cols = self.elevation_data.shape
        features = []
        valid_indices = []

        half_window = window_size // 2

        for i in range(half_window, rows - half_window):
            for j in range(half_window, cols - half_window):
                center_val = self.elevation_data[i, j]

                if np.isnan(center_val):
                    continue

                window = self.elevation_data[i-half_window:i+half_window+1, 
                                             j-half_window:j+half_window+1]

                valid_window = window[~np.isnan(window)]
                if len(valid_window) < (window_size * window_size) * 0.5:
                    continue

                feat = [center_val]
                feat.extend([
                    np.nanmean(valid_window),
                    np.nanstd(valid_window),
                    np.nanmin(valid_window),
                    np.nanmax(valid_window)
                ])
                feat.append(np.nanmax(valid_window) - np.nanmin(valid_window))

                if i > 0 and i < rows-1:
                    grad_y = abs(self.elevation_data[i+1, j] - self.elevation_data[i-1, j]) / 2
                    feat.append(grad_y if not np.isnan(grad_y) else 0)
                else:
                    feat.append(0)

                if j > 0 and j < cols-1:
                    grad_x = abs(self.elevation_data[i, j+1] - self.elevation_data[i, j-1]) / 2
                    feat.append(grad_x if not np.isnan(grad_x) else 0)
                else:
                    feat.append(0)

                local_mean = np.nanmean(valid_window)
                feat.append(abs(center_val - local_mean))

                if window_size >= 3:
                    grad_window_x = np.gradient(window, axis=1)
                    grad_window_y = np.gradient(window, axis=0)
                    roughness = np.nanstd(np.sqrt(grad_window_x**2 + grad_window_y**2))
                    feat.append(roughness if not np.isnan(roughness) else 0)
                else:
                    feat.append(0)

                if self.uncertainty_data is not None:
                    uncertainty_val = self.uncertainty_data[i, j]
                    feat.append(uncertainty_val if not np.isnan(uncertainty_val) else 0)

                features.append(feat)
                valid_indices.append((i, j))

        features_array = np.array(features)
        print(f"Extracted {len(features)} feature vectors with {features_array.shape[1]} features each")

        return features_array, valid_indices

    def detect_isolation_forest_anomalies(self, contamination: float = 0.1, window_size: int = 5) -> np.ndarray:
        """Detect anomalies using Isolation Forest."""
        print("Running Isolation Forest anomaly detection...")
        features, valid_indices = self.extract_features(window_size)

        if len(features) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        predictions = iso_forest.fit_predict(features_scaled)

        anomalies = np.zeros_like(self.elevation_data, dtype=bool)
        for idx, (i, j) in enumerate(valid_indices):
            if predictions[idx] == -1:
                anomalies[i, j] = True

        anomaly_count = np.sum(anomalies)
        print(f"Isolation Forest detected: {anomaly_count} points ({anomaly_count/anomalies.size*100:.2f}%)")

        return anomalies

    def detect_local_outlier_factor_anomalies(self, n_neighbors: int = 20, window_size: int = 5) -> np.ndarray:
        """Detect anomalies using Local Outlier Factor."""
        print("Running Local Outlier Factor anomaly detection...")
        features, valid_indices = self.extract_features(window_size)

        if len(features) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        n_neighbors = min(n_neighbors, len(features) - 1)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination='auto')
        predictions = lof.fit_predict(features_scaled)

        anomalies = np.zeros_like(self.elevation_data, dtype=bool)
        for idx, (i, j) in enumerate(valid_indices):
            if predictions[idx] == -1:
                anomalies[i, j] = True

        anomaly_count = np.sum(anomalies)
        print(f"LOF detected: {anomaly_count} points ({anomaly_count/anomalies.size*100:.2f}%)")

        return anomalies

    def detect_one_class_svm_anomalies(self, nu: float = 0.1, window_size: int = 5) -> np.ndarray:
        """Detect anomalies using One-Class SVM."""
        print("Running One-Class SVM anomaly detection...")
        features, valid_indices = self.extract_features(window_size)

        if len(features) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        svm = OneClassSVM(nu=nu, kernel='rbf', gamma='scale')
        predictions = svm.fit_predict(features_scaled)

        anomalies = np.zeros_like(self.elevation_data, dtype=bool)
        for idx, (i, j) in enumerate(valid_indices):
            if predictions[idx] == -1:
                anomalies[i, j] = True

        anomaly_count = np.sum(anomalies)
        print(f"One-Class SVM detected: {anomaly_count} points ({anomaly_count/anomalies.size*100:.2f}%)")

        return anomalies

    def detect_dbscan_anomalies(self, eps: float = 0.5, min_samples: int = 5, window_size: int = 5) -> np.ndarray:
        """Detect anomalies using DBSCAN clustering."""
        print("Running DBSCAN clustering for anomaly detection...")
        features, valid_indices = self.extract_features(window_size)

        if len(features) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)

        anomaly_indices = np.where(cluster_labels == -1)[0]

        anomalies = np.zeros_like(self.elevation_data, dtype=bool)
        for idx in anomaly_indices:
            i, j = valid_indices[idx]
            anomalies[i, j] = True

        anomaly_count = np.sum(anomalies)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        print(f"DBSCAN found {n_clusters} clusters and {anomaly_count} noise points ({anomaly_count/anomalies.size*100:.2f}%)")

        return anomalies

    def detect_pca_anomalies(self, n_components: int = 5, threshold_percentile: int = 95, window_size: int = 5) -> np.ndarray:
        """Detect anomalies using PCA reconstruction error."""
        print("Running PCA-based anomaly detection...")
        features, valid_indices = self.extract_features(window_size)

        if len(features) == 0:
            return np.zeros_like(self.elevation_data, dtype=bool)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        n_components = min(n_components, features_scaled.shape[1], features_scaled.shape[0])

        pca = PCA(n_components=n_components)
        features_pca = pca.fit_transform(features_scaled)
        features_reconstructed = pca.inverse_transform(features_pca)

        reconstruction_errors = np.sum((features_scaled - features_reconstructed) ** 2, axis=1)
        threshold = np.percentile(reconstruction_errors, threshold_percentile)

        anomalies = np.zeros_like(self.elevation_data, dtype=bool)
        for idx, (i, j) in enumerate(valid_indices):
            if reconstruction_errors[idx] > threshold:
                anomalies[i, j] = True

        anomaly_count = np.sum(anomalies)
        print(f"PCA detected: {anomaly_count} points ({anomaly_count/anomalies.size*100:.2f}%)")
        print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

        return anomalies

    def get_anomalies_for_method(self, method_name: str, **kwargs) -> Tuple[np.ndarray, str]:
        """Get anomalies for a specific method with given parameters."""
        try:
            if method_name == 'statistical':
                z_threshold = kwargs.get('z_threshold', 3.0)
                return self.detect_statistical_anomalies(z_threshold), f"Statistical (Z={z_threshold})"

            elif method_name == 'gradient':
                return self.detect_gradient_anomalies(), "Gradient"

            elif method_name == 'morphological':
                structure_size = kwargs.get('structure_size', 3)
                return self.detect_morphological_anomalies(structure_size), f"Morphological (size={structure_size})"

            elif method_name == 'uncertainty':
                result = self.detect_uncertainty_anomalies()
                if result is not None:
                    return result, "Uncertainty"
                else:
                    return np.zeros_like(self.elevation_data, dtype=bool), "Uncertainty (No Data)"

            elif method_name == 'isolation_forest':
                contamination = kwargs.get('contamination', 0.1)
                window_size = kwargs.get('window_size', 5)
                return self.detect_isolation_forest_anomalies(contamination, window_size), f"Isolation Forest (cont={contamination:.2f})"

            elif method_name == 'lof':
                n_neighbors = kwargs.get('n_neighbors', 20)
                window_size = kwargs.get('window_size', 5)
                return self.detect_local_outlier_factor_anomalies(n_neighbors, window_size), f"LOF (neighbors={n_neighbors})"

            elif method_name == 'one_class_svm':
                nu = kwargs.get('nu', 0.1)
                window_size = kwargs.get('window_size', 5)
                return self.detect_one_class_svm_anomalies(nu, window_size), f"One-Class SVM (nu={nu:.2f})"

            elif method_name == 'dbscan':
                eps = kwargs.get('eps', 0.5)
                min_samples = kwargs.get('min_samples', 5)
                window_size = kwargs.get('window_size', 5)
                return self.detect_dbscan_anomalies(eps, min_samples, window_size), f"DBSCAN (eps={eps:.2f})"

            elif method_name == 'pca':
                n_components = kwargs.get('n_components', 5)
                threshold_percentile = kwargs.get('threshold_percentile', 95)
                window_size = kwargs.get('window_size', 5)
                return self.detect_pca_anomalies(n_components, threshold_percentile, window_size), f"PCA (comp={n_components})"

            else:
                return np.zeros_like(self.elevation_data, dtype=bool), "Unknown Method"

        except Exception as e:
            print(f"Error in {method_name}: {str(e)}")
            return np.zeros_like(self.elevation_data, dtype=bool), f"{method_name} (Error)"

    def create_3d_plot(self, anomaly_mask: np.ndarray, method_name: str = "Anomaly Detection", subsample_factor: int = 1) -> go.Figure:
        """Create a 3D plot for the dashboard."""
        if self.elevation_data is None:
            return go.Figure()

        rows, cols = self.elevation_data.shape
        x_coords, y_coords = np.meshgrid(np.arange(cols), np.arange(rows))

        x_flat = x_coords.flatten()[::subsample_factor]
        y_flat = y_coords.flatten()[::subsample_factor]
        z_flat = self.elevation_data.flatten()[::subsample_factor]
        anomaly_flat = anomaly_mask.flatten()[::subsample_factor]

        valid_mask = ~np.isnan(z_flat)
        x_valid = x_flat[valid_mask]
        y_valid = y_flat[valid_mask]
        z_valid = z_flat[valid_mask]
        anomaly_valid = anomaly_flat[valid_mask]

        if len(x_valid) == 0:
            return go.Figure()

        normal_mask = ~anomaly_valid
        anomaly_points_mask = anomaly_valid

        fig = go.Figure()

        # Add normal points
        if np.any(normal_mask):
            fig.add_trace(go.Scatter3d(
                x=x_valid[normal_mask],
                y=y_valid[normal_mask],
                z=z_valid[normal_mask],
                mode='markers',
                marker=dict(
                    size=3,
                    color=z_valid[normal_mask],
                    colorscale='Viridis',
                    colorbar=dict(title="Elevation (m)", x=0.85),
                    opacity=0.7
                ),
                name='Normal Points',
                text=[f'X: {x:.0f}, Y: {y:.0f}, Z: {z:.2f}' 
                      for x, y, z in zip(x_valid[normal_mask], y_valid[normal_mask], z_valid[normal_mask])],
                hovertemplate='<b>Normal Point</b><br>%{text}<extra></extra>'
            ))

        # Add anomalous points
        if np.any(anomaly_points_mask):
            fig.add_trace(go.Scatter3d(
                x=x_valid[anomaly_points_mask],
                y=y_valid[anomaly_points_mask],
                z=z_valid[anomaly_points_mask],
                mode='markers',
                marker=dict(
                    size=6,
                    color='red',
                    symbol='diamond',
                    opacity=0.9
                ),
                name='Anomalies',
                text=[f'X: {x:.0f}, Y: {y:.0f}, Z: {z:.2f}<br>Method: {method_name}' 
                      for x, y, z in zip(x_valid[anomaly_points_mask], y_valid[anomaly_points_mask], z_valid[anomaly_points_mask])],
                hovertemplate='<b>ANOMALY</b><br>%{text}<extra></extra>'
            ))

        fig.update_layout(
            title=dict(text=f"3D Bathymetric Point Cloud - {method_name}", x=0.5, font=dict(size=16)),
            scene=dict(
                xaxis_title='X (pixels)',
                yaxis_title='Y (pixels)',
                zaxis_title='Elevation (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            width=800,
            height=600,
            showlegend=True,
            legend=dict(x=0.02, y=0.98),
            margin=dict(l=0, r=0, t=40, b=0)
        )

        return fig

    def save_anomalies_to_geotiff(self, anomaly_mask: np.ndarray, output_path: str):
        """Save anomaly detection results to a GeoTIFF file."""
        if self.dataset is None:
            raise ValueError("Original dataset not loaded")

        driver = gdal.GetDriverByName('GTiff')
        output_dataset = driver.Create(
            output_path,
            self.dataset.RasterXSize,
            self.dataset.RasterYSize,
            1,
            gdal.GDT_Byte
        )

        output_dataset.SetGeoTransform(self.dataset.GetGeoTransform())
        output_dataset.SetProjection(self.dataset.GetProjection())

        output_band = output_dataset.GetRasterBand(1)
        output_band.WriteArray(anomaly_mask.astype(np.uint8))
        output_band.SetNoDataValue(255)

        output_dataset = None
        print(f"Anomaly results saved to: {output_path}")

    def close(self):
        """Close the dataset."""
        if self.dataset:
            self.dataset = None
