import os
from osgeo import gdal
import numpy as np
import argparse
import base64
import tempfile
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from classic import BAGAnomalyDetector

gdal.UseExceptions()
detector = None


def create_dashboard(initial_bag_file_path: str = None):
    """Create and run the interactive Plotly Dash dashboard."""
    global detector
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H1("BAG Bathymetric Anomaly Detection Dashboard", 
                       className="text-center mb-4"),
                html.Hr()
            ])
        ]),

        # File Upload Section
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Load BAG File"),
                    dbc.CardBody([
                        dcc.Upload(
                            id='upload-bag',
                            children=html.Div([
                                html.I(className="fas fa-cloud-upload-alt", style={'font-size': '48px', 'color': '#007bff'}),
                                html.Br(),
                                html.Strong('Drag & Drop or Click to Select BAG File'),
                                html.Br(),
                                html.Small('Supports .bag files', className='text-muted')
                            ]),
                            style={
                                'textAlign': 'center',
                                'margin': '10px',
                                'cursor': 'pointer'
                            },
                            multiple=False
                        ),
                        html.Div(id='upload-status', className="mt-2"),
                        html.Div(id='file-info', className="mt-2")
                    ])
                ])
            ])
        ], className="mb-4"),

        html.Div(id='main-content', children=[
            dbc.Row([
                # Control Panel
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Detection Method & Parameters"),
                        dbc.CardBody([
                            # Method Selection
                            html.Label("Detection Method:", className="fw-bold"),
                            dcc.Dropdown(
                                id='method-dropdown',
                                options=[
                                    {'label': 'Statistical (Z-score)', 'value': 'statistical'},
                                    {'label': 'Gradient', 'value': 'gradient'},
                                    {'label': 'Morphological', 'value': 'morphological'},
                                    {'label': 'Uncertainty', 'value': 'uncertainty'},
                                    {'label': 'Isolation Forest', 'value': 'isolation_forest'},
                                    {'label': 'Local Outlier Factor', 'value': 'lof'},
                                    {'label': 'One-Class SVM', 'value': 'one_class_svm'},
                                    {'label': 'DBSCAN', 'value': 'dbscan'},
                                    {'label': 'PCA Reconstruction', 'value': 'pca'}
                                ],
                                value='isolation_forest',
                                className="mb-3"
                            ),

                            html.Hr(),

                            html.Div(id='traditional-params', children=[
                                html.Label("Z-Score Threshold:", className="fw-bold"),
                                dcc.Slider(
                                    id='z-threshold-slider',
                                    min=1.0, max=5.0, step=0.1, value=3.0,
                                    marks={i: str(i) for i in range(1, 6)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Br(),

                                html.Label("Morphological Structure Size:", className="fw-bold"),
                                dcc.Slider(
                                    id='structure-size-slider',
                                    min=3, max=9, step=2, value=3,
                                    marks={i: str(i) for i in range(3, 10, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ]),

                            html.Hr(),

                            html.Div(id='ml-params', children=[
                                html.Label("Contamination Rate:", className="fw-bold"),
                                dcc.Slider(
                                    id='contamination-slider',
                                    min=0.01, max=0.3, step=0.01, value=0.1,
                                    marks={0.05*i: f'{0.05*i:.2f}' for i in range(1, 7)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Br(),

                                html.Label("Feature Window Size:", className="fw-bold"),
                                dcc.Slider(
                                    id='window-size-slider',
                                    min=3, max=9, step=2, value=5,
                                    marks={i: str(i) for i in range(3, 10, 2)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Br(),

                                html.Label("LOF Neighbors:", className="fw-bold"),
                                dcc.Slider(
                                    id='n-neighbors-slider',
                                    min=5, max=50, step=5, value=20,
                                    marks={i: str(i) for i in range(5, 51, 10)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Br(),

                                html.Label("DBSCAN Epsilon:", className="fw-bold"),
                                dcc.Slider(
                                    id='eps-slider',
                                    min=0.1, max=2.0, step=0.1, value=0.5,
                                    marks={0.5*i: f'{0.5*i:.1f}' for i in range(1, 5)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                                html.Br(),

                                html.Label("PCA Components:", className="fw-bold"),
                                dcc.Slider(
                                    id='n-components-slider',
                                    min=2, max=10, step=1, value=5,
                                    marks={i: str(i) for i in range(2, 11)},
                                    tooltip={"placement": "bottom", "always_visible": True}
                                ),
                            ]),

                            html.Hr(),
                            html.Label("Subsample Factor:", className="fw-bold"),
                            dcc.Slider(
                                id='subsample-slider',
                                min=1, max=10, step=1, value=1,
                                marks={i: str(i) for i in range(1, 11)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            ),
                            html.Small("Higher values = faster rendering, fewer points", 
                                     className="text-muted"),

                            html.Hr(),
                            dbc.Button(
                                "Update Visualization", 
                                id='update-button', 
                                color="primary", 
                                size="lg",
                                className="w-100"
                            ),

                            html.Hr(),
                            html.Div(id='stats-display', className="mt-3")
                        ])
                    ])
                ], width=4),

                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("3D Point Cloud Visualization"),
                        dbc.CardBody([
                            dcc.Loading(
                                id="loading",
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id='3d-plot',
                                        style={'height': '70vh'}
                                    )
                                ]
                            )
                        ])
                    ])
                ], width=8)
            ])
        ], style={'display': 'none'})  # Hidden until file is loaded
    ], fluid=True)

    app.layout.children.append(dcc.Store(id='detector-store'))

    @app.callback(
        [Output('upload-status', 'children'),
         Output('file-info', 'children'),
         Output('main-content', 'style'),
         Output('detector-store', 'data')],
        [Input('upload-bag', 'contents')],
        [State('upload-bag', 'filename')]
    )
    def handle_file_upload(contents, filename):
        global detector

        if contents is None:
            if initial_bag_file_path and os.path.exists(initial_bag_file_path):
                detector = BAGAnomalyDetector(initial_bag_file_path)
                if detector.load_bag_file():
                    status = dbc.Alert("Initial file loaded successfully!", color="success")
                    info = dbc.Alert([
                        html.Strong(f"{os.path.basename(initial_bag_file_path)}"), html.Br(),
                        f"Size: {detector.dataset.RasterXSize} x {detector.dataset.RasterYSize}", html.Br(),
                        f"Elevation: {np.nanmin(detector.elevation_data):.1f}m to {np.nanmax(detector.elevation_data):.1f}m",
                        html.Br() if detector.uncertainty_data is not None else "",
                        f"Uncertainty: {np.nanmin(detector.uncertainty_data):.1f}m to {np.nanmax(detector.uncertainty_data):.1f}m" if detector.uncertainty_data is not None else ""
                    ], color="info")
                    return status, info, {'display': 'block'}, {'file_loaded': True}
            return "", "", {'display': 'none'}, {'file_loaded': False}

        try:
            content_type, content_string = contents.split(',')
            decoded = base64.b64decode(content_string)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.bag') as tmp_file:
                tmp_file.write(decoded)
                tmp_path = tmp_file.name
            detector = BAGAnomalyDetector(tmp_path)
            if detector.load_bag_file():
                status = dbc.Alert("File uploaded and loaded successfully!", color="success")
                info = dbc.Alert([
                    html.Strong(f"{filename}"), html.Br(),
                    f"Size: {detector.dataset.RasterXSize} x {detector.dataset.RasterYSize}", html.Br(),
                    f"Elevation: {np.nanmin(detector.elevation_data):.1f}m to {np.nanmax(detector.elevation_data):.1f}m",
                    html.Br() if detector.uncertainty_data is not None else "",
                    f"Uncertainty: {np.nanmin(detector.uncertainty_data):.1f}m to {np.nanmax(detector.uncertainty_data):.1f}m" if detector.uncertainty_data is not None else ""
                ], color="info")

                return status, info, {'display': 'block'}, {'file_loaded': True, 'temp_path': tmp_path}
            else:
                status = dbc.Alert("Failed to load BAG file. Please check the file format.", color="danger")
                return status, "", {'display': 'none'}, {'file_loaded': False}

        except Exception as e:
            status = dbc.Alert(f"Error processing file: {str(e)}", color="danger")
            return status, "", {'display': 'none'}, {'file_loaded': False}

    @app.callback(
        [Output('3d-plot', 'figure'),
         Output('stats-display', 'children')],
        [Input('update-button', 'n_clicks')],
        [State('method-dropdown', 'value'),
         State('z-threshold-slider', 'value'),
         State('structure-size-slider', 'value'),
         State('contamination-slider', 'value'),
         State('window-size-slider', 'value'),
         State('n-neighbors-slider', 'value'),
         State('eps-slider', 'value'),
         State('n-components-slider', 'value'),
         State('subsample-slider', 'value'),
         State('detector-store', 'data')]
    )
    def update_visualization(n_clicks, method, z_threshold, structure_size, 
                           contamination, window_size, n_neighbors, eps, 
                           n_components, subsample, store_data):

        if store_data is None or not store_data.get('file_loaded', False):
            return go.Figure(), dbc.Alert("Please upload the input file and click on the Visualization button", color="warning")

        if detector is None:
            return go.Figure(), dbc.Alert("No data loaded", color="danger")

        try:
            params = {
                'z_threshold': z_threshold,
                'structure_size': structure_size,
                'contamination': contamination,
                'window_size': window_size,
                'n_neighbors': n_neighbors,
                'eps': eps,
                'min_samples': 5,
                'n_components': n_components,
                'threshold_percentile': 95,
                'nu': contamination
            }

            anomaly_mask, method_display_name = detector.get_anomalies_for_method(method, **params)
            fig = detector.create_3d_plot(anomaly_mask, method_display_name, subsample)
            total_points = anomaly_mask.size
            anomaly_points = np.sum(anomaly_mask)
            anomaly_percentage = (anomaly_points / total_points) * 100
            stats = dbc.Alert([
                html.H5("Detection Statistics", className="alert-heading"),
                html.P([
                    html.Strong("Method: "), method_display_name, html.Br(),
                    html.Strong("Total Points: "), f"{total_points:,}", html.Br(),
                    html.Strong("Anomalies Found: "), f"{anomaly_points:,}", html.Br(),
                    html.Strong("Anomaly Rate: "), f"{anomaly_percentage:.2f}%"
                ]),
                html.Hr(),
                html.P("Red diamonds = Detected anomalies", className="mb-0"),
                html.P("Blue points = Normal bathymetry", className="mb-0")
            ], color="info")

            return fig, stats

        except Exception as e:
            error_msg = dbc.Alert(f"Error during analysis: {str(e)}", color="danger")
            return go.Figure(), error_msg

    @app.callback(
        [Output('traditional-params', 'style'),
         Output('ml-params', 'style')],
        [Input('method-dropdown', 'value')]
    )
    def toggle_params(method):
        traditional_methods = ['statistical', 'gradient', 'morphological', 'uncertainty']
        ml_methods = ['isolation_forest', 'lof', 'one_class_svm', 'dbscan', 'pca']

        if method in traditional_methods:
            return {'display': 'block'}, {'display': 'none'}
        elif method in ml_methods:
            return {'display': 'none'}, {'display': 'block'}
        else:
            return {'display': 'block'}, {'display': 'block'}

    print("Starting Bathymetric Anomaly Detection Dashboard...")
    app.run(debug=True, port=8050)

def main():
    """Main function with both command line and dashboard options."""
    parser = argparse.ArgumentParser(description='BAG Bathymetric Anomaly Detection')
    parser.add_argument('bag_file', nargs='?', help='Path to the BAG file (optional for dashboard mode)')
    parser.add_argument('--dashboard', action='store_true', 
                       help='Launch interactive dashboard (default: command line mode)')
    parser.add_argument('--output', '-o', help='Output GeoTIFF file path (command line mode)')
    parser.add_argument('--method', choices=['statistical', 'gradient', 'morphological', 'uncertainty',
                                            'isolation_forest', 'lof', 'one_class_svm', 'dbscan', 'pca'], 
                       default='isolation_forest', help='Detection method (command line mode)')
    parser.add_argument('--z-threshold', type=float, default=3.0, help='Z-score threshold')
    parser.add_argument('--contamination', type=float, default=0.1, help='ML contamination rate')
    parser.add_argument('--window-size', type=int, default=5, help='Feature extraction window size')

    args = parser.parse_args()

    if args.dashboard:
        create_dashboard(args.bag_file)
    else:
        if not args.bag_file:
            print("Error: BAG file required for command line mode")
            print("Usage: python anomaly.py <bag_file> [options]")
            print("   or: python anomaly.py --dashboard  (for upload interface)")
            return
        if not os.path.exists(args.bag_file):
            print(f"Error: BAG file not found: {args.bag_file}")
            return
        detector = BAGAnomalyDetector(args.bag_file)
        if not detector.load_bag_file():
            return
        print(f"Running {args.method} anomaly detection...")
        params = {
            'z_threshold': args.z_threshold,
            'contamination': args.contamination,
            'window_size': args.window_size,
            'n_neighbors': 20,
            'eps': 0.5,
            'min_samples': 5,
            'n_components': 5,
            'nu': args.contamination,
            'structure_size': 3
        }

        anomaly_mask, method_name = detector.get_anomalies_for_method(args.method, **params)
        fig = detector.create_3d_plot(anomaly_mask, method_name)
        fig.show()
        if args.output:
            try:
                detector.save_anomalies_to_geotiff(anomaly_mask, args.output)
                print(f"Results saved to: {args.output}")
            except Exception as e:
                print(f"Error saving results: {e}")

        total_points = anomaly_mask.size
        anomaly_points = np.sum(anomaly_mask)
        print("\nSummary:")
        print(f"Method: {method_name}")
        print(f"Total points: {total_points:,}")
        print(f"Anomalous points: {anomaly_points:,}")
        print(f"Anomaly percentage: {anomaly_points/total_points*100:.2f}%")
        detector.close()


if __name__ == "__main__":
    main()
