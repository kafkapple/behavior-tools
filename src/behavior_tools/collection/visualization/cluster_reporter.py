"""
Cluster visualization with thumbnails.
"""
from pathlib import Path
from typing import List, Dict
import base64
import logging

logger = logging.getLogger(__name__)


class ClusterReporter:
    """Generate HTML reports with cluster visualizations."""

    def __init__(self, output_dir: Path):
        """Initialize reporter."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _image_to_base64(self, image_path: Path, max_size: int = 200) -> str:
        """Convert image to base64 thumbnail."""
        try:
            from PIL import Image
            import io

            img = Image.open(image_path)

            # Convert to RGB if necessary (for RGBA, P, LA modes)
            if img.mode in ('RGBA', 'LA', 'P'):
                # Create white background
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    background.paste(img, mask=img.split()[3])  # Use alpha channel as mask
                else:
                    background.paste(img)
                img = background

            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=85)
            img_str = base64.b64encode(buffer.getvalue()).decode()

            return f"data:image/jpeg;base64,{img_str}"

        except Exception as e:
            logger.error(f"Error converting {image_path} to base64: {e}")
            return ""

    def generate_cluster_report(
        self,
        cluster_data: Dict[int, List[Path]],
        cluster_stats: Dict = None,
        experiment_name: str = "cluster_analysis"
    ):
        """
        Generate HTML report with cluster thumbnails.

        Args:
            cluster_data: {cluster_id: [image_paths]}
            cluster_stats: Optional statistics
            experiment_name: Name for the report
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Cluster Analysis - {experiment_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        .summary {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .summary-stat {{
            display: inline-block;
            margin-right: 30px;
            font-size: 18px;
        }}
        .summary-stat .value {{
            font-weight: bold;
            color: #4CAF50;
            font-size: 24px;
        }}
        .cluster {{
            background: white;
            margin-bottom: 30px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .cluster-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 20px;
            font-weight: bold;
            color: #333;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid #ddd;
        }}
        .cluster-controls {{
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        .btn-include, .btn-exclude {{
            padding: 8px 16px;
            border: 2px solid #ddd;
            border-radius: 5px;
            background: white;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }}
        .btn-include:hover {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        .btn-exclude:hover {{
            background: #f44336;
            color: white;
            border-color: #f44336;
        }}
        .btn-include.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}
        .btn-exclude.active {{
            background: #f44336;
            color: white;
            border-color: #f44336;
        }}
        .cluster-status {{
            font-size: 20px;
            margin-left: 10px;
        }}
        .cluster-status.included {{
            color: #4CAF50;
        }}
        .cluster-status.excluded {{
            color: #f44336;
        }}
        .cluster.dimmed {{
            opacity: 0.3;
        }}
        .export-panel {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.2);
            z-index: 1000;
        }}
        .export-panel h3 {{
            margin-top: 0;
            font-size: 18px;
        }}
        .export-btn {{
            padding: 12px 24px;
            background: #2196F3;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 10px;
        }}
        .export-btn:hover {{
            background: #1976D2;
        }}
        .selection-count {{
            font-size: 14px;
            color: #666;
            margin-top: 10px;
        }}
        .cluster-info {{
            color: #666;
            margin-bottom: 15px;
            font-size: 14px;
        }}
        .thumbnails {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
            gap: 15px;
        }}
        .thumbnail {{
            text-align: center;
        }}
        .thumbnail img {{
            width: 100%;
            height: 150px;
            object-fit: contain;
            background: #f0f0f0;
            border-radius: 4px;
            cursor: pointer;
            transition: transform 0.2s;
        }}
        .thumbnail img:hover {{
            transform: scale(1.05);
            box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        }}
        .thumbnail .filename {{
            font-size: 11px;
            color: #666;
            margin-top: 5px;
            word-break: break-all;
        }}
        .modal {{
            display: none;
            position: fixed;
            z-index: 1000;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0,0,0,0.9);
        }}
        .modal-content {{
            margin: auto;
            display: block;
            max-width: 90%;
            max-height: 90%;
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
        }}
        .close {{
            position: absolute;
            top: 15px;
            right: 35px;
            color: #f1f1f1;
            font-size: 40px;
            font-weight: bold;
            cursor: pointer;
        }}
        .close:hover {{
            color: #bbb;
        }}
    </style>
</head>
<body>
    <h1>Cluster Analysis: {experiment_name}</h1>
"""

        # Summary section
        total_clusters = len(cluster_data)
        total_images = sum(len(paths) for paths in cluster_data.values())

        html += f"""
    <div class="summary">
        <div class="summary-stat">
            <div class="label">Total Clusters:</div>
            <div class="value">{total_clusters}</div>
        </div>
        <div class="summary-stat">
            <div class="label">Total Images:</div>
            <div class="value">{total_images}</div>
        </div>
        <div class="summary-stat">
            <div class="label">Avg per Cluster:</div>
            <div class="value">{total_images/total_clusters:.1f}</div>
        </div>
    </div>
"""

        # Cluster sections
        for cluster_id in sorted(cluster_data.keys()):
            paths = cluster_data[cluster_id]

            html += f"""
    <div class="cluster" data-cluster-id="{cluster_id}">
        <div class="cluster-header">
            <span>Cluster #{cluster_id}</span>
            <div class="cluster-controls">
                <button class="btn-include" onclick="toggleCluster({cluster_id}, 'include')">✓ Include</button>
                <button class="btn-exclude" onclick="toggleCluster({cluster_id}, 'exclude')">✗ Exclude</button>
                <span class="cluster-status" id="status-{cluster_id}">●</span>
            </div>
        </div>
        <div class="cluster-info">
            {len(paths)} images
        </div>
        <div class="thumbnails">
"""

            # Add thumbnails
            for img_path in paths:
                img_base64 = self._image_to_base64(img_path)
                if img_base64:
                    filename = Path(img_path).name
                    html += f"""
            <div class="thumbnail">
                <img src="{img_base64}" alt="{filename}" onclick="openModal(this.src)">
                <div class="filename">{filename}</div>
            </div>
"""

            html += """
        </div>
    </div>
"""

        # Modal for full-size viewing
        html += """
    <div id="imageModal" class="modal" onclick="closeModal()">
        <span class="close">&times;</span>
        <img class="modal-content" id="modalImage">
    </div>

    <!-- Export Panel -->
    <div class="export-panel">
        <h3>Cluster Selection</h3>
        <div class="selection-count">
            Included: <span id="included-count">0</span><br>
            Excluded: <span id="excluded-count">0</span><br>
            Neutral: <span id="neutral-count">""" + str(total_clusters) + """</span>
        </div>
        <button class="export-btn" onclick="exportSelections()">📥 Export Selections</button>
        <button class="export-btn" onclick="resetSelections()" style="background: #9E9E9E; margin-top: 5px;">🔄 Reset All</button>
    </div>

    <script>
        // Cluster selection state
        let clusterSelections = {};

        function toggleCluster(clusterId, action) {
            const cluster = document.querySelector(`[data-cluster-id="${clusterId}"]`);
            const status = document.getElementById(`status-${clusterId}`);
            const includeBtn = cluster.querySelector('.btn-include');
            const excludeBtn = cluster.querySelector('.btn-exclude');

            // Toggle state
            if (clusterSelections[clusterId] === action) {
                // Deselect
                delete clusterSelections[clusterId];
                cluster.classList.remove('dimmed');
                status.className = 'cluster-status';
                includeBtn.classList.remove('active');
                excludeBtn.classList.remove('active');
            } else {
                // Select
                clusterSelections[clusterId] = action;

                if (action === 'include') {
                    cluster.classList.remove('dimmed');
                    status.className = 'cluster-status included';
                    includeBtn.classList.add('active');
                    excludeBtn.classList.remove('active');
                } else {
                    cluster.classList.add('dimmed');
                    status.className = 'cluster-status excluded';
                    excludeBtn.classList.add('active');
                    includeBtn.classList.remove('active');
                }
            }

            updateSelectionCounts();
        }

        function updateSelectionCounts() {
            const totalClusters = """ + str(total_clusters) + """;
            const included = Object.values(clusterSelections).filter(a => a === 'include').length;
            const excluded = Object.values(clusterSelections).filter(a => a === 'exclude').length;
            const neutral = totalClusters - included - excluded;

            document.getElementById('included-count').textContent = included;
            document.getElementById('excluded-count').textContent = excluded;
            document.getElementById('neutral-count').textContent = neutral;
        }

        function resetSelections() {
            clusterSelections = {};

            // Reset all clusters
            document.querySelectorAll('.cluster').forEach(cluster => {
                cluster.classList.remove('dimmed');
                cluster.querySelector('.cluster-status').className = 'cluster-status';
                cluster.querySelector('.btn-include').classList.remove('active');
                cluster.querySelector('.btn-exclude').classList.remove('active');
            });

            updateSelectionCounts();
        }

        function exportSelections() {
            const included = [];
            const excluded = [];

            for (const [clusterId, action] of Object.entries(clusterSelections)) {
                if (action === 'include') {
                    included.push(parseInt(clusterId));
                } else if (action === 'exclude') {
                    excluded.push(parseInt(clusterId));
                }
            }

            const data = {
                timestamp: new Date().toISOString(),
                included_clusters: included,
                excluded_clusters: excluded,
                total_clusters: """ + str(total_clusters) + """,
                mode: included.length > 0 ? 'include' : (excluded.length > 0 ? 'exclude' : 'none')
            };

            // Create download
            const blob = new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'cluster_selections.json';
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);

            alert('Selections exported!\\n\\nIncluded: ' + included.length + ' clusters\\nExcluded: ' + excluded.length + ' clusters\\n\\nNext step: Run process_cluster_selections.py');
        }

        function openModal(src) {
            document.getElementById('imageModal').style.display = 'block';
            document.getElementById('modalImage').src = src;
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        // Close on Esc key
        document.addEventListener('keydown', function(event) {
            if (event.key === 'Escape') {
                closeModal();
            }
        });

        // Initialize counts
        updateSelectionCounts();
    </script>
</body>
</html>
"""

        # Save report
        report_path = self.output_dir / f"{experiment_name}_cluster_report.html"
        report_path.write_text(html)

        logger.info(f"✓ Cluster report saved: {report_path}")
        return report_path
