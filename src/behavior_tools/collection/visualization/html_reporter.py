"""
HTML report generator for visual inspection of curated datasets.
Creates interactive reports with images grouped by cluster, filtering reason, etc.
"""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import json
import base64
from io import BytesIO
from PIL import Image
import shutil

logger = logging.getLogger(__name__)


class HTMLReporter:
    """Generate HTML reports for dataset inspection."""

    def __init__(self, output_dir: Path):
        """
        Initialize HTML reporter.

        Args:
            output_dir: Directory to save HTML reports
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def image_to_base64(self, image_path: Path, max_size: int = 300) -> str:
        """
        Convert image to base64 for embedding in HTML.

        Args:
            image_path: Path to image
            max_size: Maximum dimension for thumbnail

        Returns:
            Base64 encoded image string
        """
        try:
            img = Image.open(image_path)
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"
        except Exception as e:
            logger.debug(f"Error converting {image_path} to base64: {e}")
            return ""

    def generate_cluster_report(
        self,
        cluster_info: Dict,
        image_paths: List[Path],
        experiment_name: str = "clustering"
    ):
        """
        Generate HTML report for clustering results.

        Args:
            cluster_info: Dictionary with cluster information
            image_paths: List of all image paths
            experiment_name: Name of the experiment
        """
        html_path = self.output_dir / f"{experiment_name}_cluster_report.html"

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cluster Report - {experiment_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #667eea;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .cluster {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .cluster-header {{
            padding: 20px;
            border-bottom: 2px solid #f0f0f0;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .cluster-header.kept {{
            background: #d4edda;
        }}
        .cluster-header.removed {{
            background: #f8d7da;
        }}
        .cluster-header h2 {{
            margin: 0;
            font-size: 20px;
        }}
        .cluster-badge {{
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: bold;
        }}
        .cluster-badge.kept {{
            background: #28a745;
            color: white;
        }}
        .cluster-badge.removed {{
            background: #dc3545;
            color: white;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 20px;
        }}
        .image-item {{
            position: relative;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            background: #fafafa;
        }}
        .image-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .image-item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: white;
        }}
        .image-info {{
            padding: 10px;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        .image-info .filename {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            word-break: break-all;
        }}
        .toggle-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            margin-left: 10px;
        }}
        .toggle-btn:hover {{
            background: #5568d3;
        }}
        .collapsed .image-grid {{
            display: none;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Cluster Analysis Report</h1>
        <p><strong>Experiment:</strong> {experiment_name}</p>
        <p><strong>Total Images:</strong> {len(image_paths)}</p>
    </div>
"""

        # Calculate statistics
        total_clusters = len(cluster_info)
        kept_clusters = sum(1 for c in cluster_info.values() if c['kept'])
        removed_clusters = total_clusters - kept_clusters
        total_kept_images = sum(c['size'] for c in cluster_info.values() if c['kept'])
        total_removed_images = sum(c['size'] for c in cluster_info.values() if not c['kept'])

        html += f"""
    <div class="stats">
        <div class="stat-card">
            <h3>Total Clusters</h3>
            <div class="value">{total_clusters}</div>
        </div>
        <div class="stat-card">
            <h3>Kept Clusters</h3>
            <div class="value" style="color: #28a745;">{kept_clusters}</div>
        </div>
        <div class="stat-card">
            <h3>Removed Clusters</h3>
            <div class="value" style="color: #dc3545;">{removed_clusters}</div>
        </div>
        <div class="stat-card">
            <h3>Kept Images</h3>
            <div class="value" style="color: #28a745;">{total_kept_images}</div>
        </div>
        <div class="stat-card">
            <h3>Removed Images</h3>
            <div class="value" style="color: #dc3545;">{total_removed_images}</div>
        </div>
    </div>
"""

        # Sort clusters by size (descending)
        sorted_clusters = sorted(cluster_info.items(), key=lambda x: x[1]['size'], reverse=True)

        for cluster_id, info in sorted_clusters:
            status = "kept" if info['kept'] else "removed"
            status_text = "KEPT" if info['kept'] else "REMOVED"

            html += f"""
    <div class="cluster">
        <div class="cluster-header {status}">
            <div>
                <h2>Cluster {cluster_id}</h2>
                <p style="margin: 5px 0 0 0; color: #666;">
                    {info['size']} images ({info['percentage']:.1f}%)
                </p>
            </div>
            <div>
                <span class="cluster-badge {status}">{status_text}</span>
                <button class="toggle-btn" onclick="toggleCluster(this)">Show/Hide</button>
            </div>
        </div>
        <div class="image-grid">
"""

            # Add up to 20 images per cluster for performance
            image_subset = info['images'][:20]
            for img_path_str in image_subset:
                img_path = Path(img_path_str)
                if img_path.exists():
                    img_base64 = self.image_to_base64(img_path)
                    if img_base64:
                        html += f"""
            <div class="image-item">
                <img src="{img_base64}" alt="{img_path.name}">
                <div class="image-info">
                    <div class="filename">{img_path.name}</div>
                    <div>{img_path.parent.name}</div>
                </div>
            </div>
"""

            if len(info['images']) > 20:
                html += f"""
            <div class="image-item" style="display: flex; align-items: center; justify-content: center; background: #f0f0f0;">
                <p style="text-align: center; color: #666;">+ {len(info['images']) - 20} more images</p>
            </div>
"""

            html += """
        </div>
    </div>
"""

        html += """
    <script>
        function toggleCluster(btn) {
            const cluster = btn.closest('.cluster');
            cluster.classList.toggle('collapsed');
        }
    </script>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Cluster report saved to: {html_path}")

    def generate_filtering_report(
        self,
        stats: Dict,
        accepted_images: List[Path],
        rejected_images: List[Path],
        experiment_name: str = "filtering"
    ):
        """
        Generate HTML report for semantic filtering results.

        Args:
            stats: Statistics dictionary with filtering details
            accepted_images: List of accepted image paths
            rejected_images: List of rejected image paths
            experiment_name: Name of the experiment
        """
        html_path = self.output_dir / f"{experiment_name}_filtering_report.html"

        # Extract rejection details
        details = stats.get('details', [])
        rejected_details = [d for d in details if not d.get('accepted', False)]

        # Group by rejection reason
        by_reason = {}
        for detail in rejected_details:
            reason = detail.get('reason', 'unknown')
            if reason not in by_reason:
                by_reason[reason] = []
            by_reason[reason].append(detail)

        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Filtering Report - {experiment_name}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0 0 10px 0;
        }}
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .tab {{
            background: white;
            border: 2px solid #e0e0e0;
            padding: 15px 25px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: bold;
        }}
        .tab:hover {{
            border-color: #f5576c;
            transform: translateY(-2px);
        }}
        .tab.active {{
            background: #f5576c;
            color: white;
            border-color: #f5576c;
        }}
        .tab-content {{
            display: none;
        }}
        .tab-content.active {{
            display: block;
        }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-card h3 {{
            margin: 0 0 10px 0;
            color: #f5576c;
            font-size: 14px;
            text-transform: uppercase;
        }}
        .stat-card .value {{
            font-size: 32px;
            font-weight: bold;
            color: #333;
        }}
        .section {{
            background: white;
            margin-bottom: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .section-header {{
            padding: 20px;
            background: #f8f9fa;
            border-bottom: 2px solid #e0e0e0;
        }}
        .section-header h2 {{
            margin: 0;
            font-size: 20px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
            gap: 15px;
            padding: 20px;
        }}
        .image-item {{
            position: relative;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
            transition: transform 0.2s, box-shadow 0.2s;
            background: #fafafa;
        }}
        .image-item:hover {{
            transform: translateY(-5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .image-item img {{
            width: 100%;
            height: 200px;
            object-fit: contain;
            background: white;
        }}
        .image-info {{
            padding: 10px;
            font-size: 12px;
            color: #666;
            border-top: 1px solid #e0e0e0;
        }}
        .image-info .filename {{
            font-weight: bold;
            color: #333;
            margin-bottom: 5px;
            word-break: break-all;
        }}
        .score {{
            display: inline-block;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
            margin-top: 5px;
        }}
        .score.high {{
            background: #d4edda;
            color: #155724;
        }}
        .score.medium {{
            background: #fff3cd;
            color: #856404;
        }}
        .score.low {{
            background: #f8d7da;
            color: #721c24;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Semantic Filtering Report</h1>
        <p><strong>Experiment:</strong> {experiment_name}</p>
        <p><strong>Strategy:</strong> {stats.get('strategy', 'unknown')}</p>
    </div>

    <div class="stats">
        <div class="stat-card">
            <h3>Total Images</h3>
            <div class="value">{stats.get('total', 0)}</div>
        </div>
        <div class="stat-card">
            <h3>Accepted</h3>
            <div class="value" style="color: #28a745;">{stats.get('accepted', 0)}</div>
        </div>
        <div class="stat-card">
            <h3>Rejected</h3>
            <div class="value" style="color: #dc3545;">{stats.get('rejected', 0)}</div>
        </div>
        <div class="stat-card">
            <h3>Acceptance Rate</h3>
            <div class="value" style="color: #667eea;">{stats.get('acceptance_rate', 0)*100:.1f}%</div>
        </div>
    </div>

    <div class="tabs">
        <div class="tab active" onclick="showTab('accepted')">✓ Accepted Images ({len(accepted_images)})</div>
"""

        # Add tabs for each rejection reason
        for reason, items in by_reason.items():
            reason_label = reason.replace('_', ' ').title()
            html += f"""        <div class="tab" onclick="showTab('{reason}')">✗ {reason_label} ({len(items)})</div>\n"""

        html += """    </div>

    <!-- Accepted Images Tab -->
    <div id="accepted" class="tab-content active">
        <div class="section">
            <div class="section-header">
                <h2>✓ Accepted Images</h2>
            </div>
            <div class="image-grid">
"""

        # Show up to 50 accepted images
        for img_path in accepted_images[:50]:
            if img_path.exists():
                img_base64 = self.image_to_base64(img_path)
                if img_base64:
                    # Find details for this image
                    img_detail = next((d for d in details if d.get('path') == str(img_path)), {})
                    positive_score = img_detail.get('max_positive_score', 0)
                    matched_concept = img_detail.get('matched_concept', '')

                    score_class = 'high' if positive_score > 0.5 else 'medium' if positive_score > 0.35 else 'low'

                    html += f"""
                <div class="image-item">
                    <img src="{img_base64}" alt="{img_path.name}">
                    <div class="image-info">
                        <div class="filename">{img_path.name}</div>
                        <div>{img_path.parent.name}</div>
                        <div class="score {score_class}">Score: {positive_score:.3f}</div>
                        <div style="font-size: 10px; margin-top: 3px; color: #888;">{matched_concept[:40]}</div>
                    </div>
                </div>
"""

        if len(accepted_images) > 50:
            html += f"""
                <div class="image-item" style="display: flex; align-items: center; justify-content: center; background: #f0f0f0;">
                    <p style="text-align: center; color: #666;">+ {len(accepted_images) - 50} more images</p>
                </div>
"""

        html += """
            </div>
        </div>
    </div>
"""

        # Rejection reason tabs
        for reason, items in by_reason.items():
            reason_label = reason.replace('_', ' ').title()

            html += f"""
    <div id="{reason}" class="tab-content">
        <div class="section">
            <div class="section-header">
                <h2>✗ Rejected: {reason_label}</h2>
                <p style="margin: 5px 0 0 0; color: #666;">{len(items)} images</p>
            </div>
            <div class="image-grid">
"""

            # Show up to 30 rejected images per reason
            for item in items[:30]:
                img_path = Path(item.get('path', ''))
                if img_path.exists():
                    img_base64 = self.image_to_base64(img_path)
                    if img_base64:
                        rejection_info = item.get('rejection_reason', '')
                        category = item.get('rejected_category', '')

                        html += f"""
                <div class="image-item">
                    <img src="{img_base64}" alt="{img_path.name}">
                    <div class="image-info">
                        <div class="filename">{img_path.name}</div>
                        <div>{img_path.parent.name}</div>
                        <div style="font-size: 10px; margin-top: 5px; color: #dc3545;">
                            {category}: {rejection_info[:60] if rejection_info else 'N/A'}
                        </div>
                    </div>
                </div>
"""

            if len(items) > 30:
                html += f"""
                <div class="image-item" style="display: flex; align-items: center; justify-content: center; background: #f0f0f0;">
                    <p style="text-align: center; color: #666;">+ {len(items) - 30} more images</p>
                </div>
"""

            html += """
            </div>
        </div>
    </div>
"""

        html += """
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));

            // Remove active class from all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));

            // Show selected tab content
            const selectedContent = document.getElementById(tabName);
            if (selectedContent) {
                selectedContent.classList.add('active');
            }

            // Highlight selected tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Filtering report saved to: {html_path}")

    def generate_comparison_report(
        self,
        experiments: Dict[str, Dict],
        output_name: str = "comparison"
    ):
        """
        Generate comparison report across multiple experiments.

        Args:
            experiments: Dict mapping experiment name to stats dict
            output_name: Name for output file
        """
        html_path = self.output_dir / f"{output_name}_report.html"

        html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Experiment Comparison Report</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        table {
            width: 100%;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-collapse: collapse;
        }
        th {
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
        }
        td {
            padding: 15px;
            border-bottom: 1px solid #e0e0e0;
        }
        tr:hover {
            background: #f8f9fa;
        }
        .best {
            background: #d4edda;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Experiment Comparison</h1>
    </div>
    <table>
        <thead>
            <tr>
                <th>Experiment</th>
                <th>Strategy</th>
                <th>Total</th>
                <th>Accepted</th>
                <th>Rejected</th>
                <th>Acceptance Rate</th>
            </tr>
        </thead>
        <tbody>
"""

        for exp_name, stats in experiments.items():
            acceptance_rate = stats.get('acceptance_rate', 0) * 100

            html += f"""
            <tr>
                <td><strong>{exp_name}</strong></td>
                <td>{stats.get('strategy', 'N/A')}</td>
                <td>{stats.get('total', 0)}</td>
                <td>{stats.get('accepted', 0)}</td>
                <td>{stats.get('rejected', 0)}</td>
                <td>{acceptance_rate:.1f}%</td>
            </tr>
"""

        html += """
        </tbody>
    </table>
</body>
</html>
"""

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html)

        logger.info(f"Comparison report saved to: {html_path}")
