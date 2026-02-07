"""
Active learning interface for uncertain images.
"""
from pathlib import Path
from typing import List, Dict
import base64
import logging

logger = logging.getLogger(__name__)


def generate_active_learning_interface(
    uncertain_images: List[tuple],  # [(path, details), ...]
    output_html: Path,
    max_images: int = 50
):
    """
    Generate interactive HTML for reviewing uncertain images.

    Args:
        uncertain_images: List of (image_path, filter_details) tuples
        output_html: Path to save HTML
        max_images: Maximum images to show
    """
    # Limit to max_images
    uncertain_images = uncertain_images[:max_images]

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Active Learning - Uncertain Images</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        .info {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            border-left: 4px solid #ffc107;
        }
        .controls {
            position: sticky;
            top: 0;
            background: white;
            padding: 15px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            z-index: 100;
        }
        .stats {
            display: inline-block;
            margin-right: 20px;
            font-weight: bold;
        }
        .accepted { color: green; }
        .rejected { color: red; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }
        .card {
            background: white;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .card.accepted-card { border: 3px solid green; }
        .card.rejected-card { border: 3px solid red; }
        .card img {
            width: 100%;
            height: 250px;
            object-fit: contain;
            background: #f0f0f0;
            border-radius: 4px;
            margin-bottom: 10px;
        }
        .scores {
            font-size: 12px;
            color: #666;
            margin-bottom: 10px;
            padding: 10px;
            background: #f9f9f9;
            border-radius: 4px;
        }
        .score-item {
            margin: 5px 0;
        }
        .score-label {
            font-weight: bold;
            display: inline-block;
            width: 120px;
        }
        .buttons { display: flex; gap: 10px; margin-top: 10px; }
        button {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
            font-weight: bold;
        }
        .accept-btn { background: #4CAF50; color: white; }
        .accept-btn:hover { background: #45a049; }
        .reject-btn { background: #f44336; color: white; }
        .reject-btn:hover { background: #da190b; }
        .export-btn {
            background: #2196F3;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            font-weight: bold;
        }
        .export-btn:hover { background: #0b7dda; }
    </style>
</head>
<body>
    <h1>Active Learning - Review Uncertain Images</h1>

    <div class="info">
        ℹ️ These images were flagged as <strong>uncertain</strong> by the automatic filters.
        Please review and accept/reject each image manually.
    </div>

    <div class="controls">
        <div>
            <span class="stats">Total: <span id="total">0</span></span>
            <span class="stats accepted">Accepted: <span id="accepted-count">0</span></span>
            <span class="stats rejected">Rejected: <span id="rejected-count">0</span></span>
            <span class="stats">Remaining: <span id="remaining">0</span></span>
        </div>
        <button class="export-btn" onclick="exportSelections()">💾 Export Selections</button>
    </div>

    <div class="grid" id="grid">
"""

    for idx, (img_path, details) in enumerate(uncertain_images):
        # Embed image as base64
        try:
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode()

            # Extract scores
            scores_html = ""
            if 'ensemble_score' in details:
                scores_html += f'<div class="score-item"><span class="score-label">Ensemble:</span> {details["ensemble_score"]:.3f}</div>'
            if 'target_score' in details:
                scores_html += f'<div class="score-item"><span class="score-label">CLIP Target:</span> {details["target_score"]:.3f}</div>'
            if 'positive_similarity' in details:
                scores_html += f'<div class="score-item"><span class="score-label">DINOv2 Positive:</span> {details["positive_similarity"]:.3f}</div>'
            if 'negative_similarity' in details:
                scores_html += f'<div class="score-item"><span class="score-label">DINOv2 Negative:</span> {details["negative_similarity"]:.3f}</div>'

            reason = details.get('reason', 'unknown')

            html += f"""
        <div class="card" id="card-{idx}">
            <img src="data:image/jpeg;base64,{img_data}" />
            <div class="scores">
                <div class="score-item"><span class="score-label">Reason:</span> {reason}</div>
                {scores_html}
            </div>
            <div class="buttons">
                <button class="accept-btn" onclick="markAs('{idx}', 'accepted', '{Path(img_path).name}')">✓ Accept</button>
                <button class="reject-btn" onclick="markAs('{idx}', 'rejected', '{Path(img_path).name}')">✗ Reject</button>
            </div>
        </div>
"""
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")

    html += f"""
    </div>

    <script>
        let selections = {{}};
        const total = {len(uncertain_images)};

        function updateStats() {{
            const accepted = Object.values(selections).filter(s => s.status === 'accepted').length;
            const rejected = Object.values(selections).filter(s => s.status === 'rejected').length;
            const remaining = total - accepted - rejected;

            document.getElementById('total').textContent = total;
            document.getElementById('accepted-count').textContent = accepted;
            document.getElementById('rejected-count').textContent = rejected;
            document.getElementById('remaining').textContent = remaining;
        }}

        function markAs(idx, status, filename) {{
            selections[idx] = {{ filename: filename, status: status }};

            const card = document.getElementById('card-' + idx);
            card.classList.remove('accepted-card', 'rejected-card');
            card.classList.add(status + '-card');

            updateStats();
        }}

        function exportSelections() {{
            if (Object.keys(selections).length === 0) {{
                alert('No selections made yet!');
                return;
            }}

            const dataStr = JSON.stringify(selections, null, 2);
            const dataBlob = new Blob([dataStr], {{type: 'application/json'}});
            const url = URL.createObjectURL(dataBlob);
            const link = document.createElement('a');
            link.href = url;
            link.download = 'active_learning_selections.json';
            link.click();

            alert('Selections exported! Process with: python process_manual_selections.py active_learning_selections.json');
        }}

        // Keyboard shortcuts
        document.addEventListener('keydown', function(event) {{
            // Space = Accept, Backspace = Reject, for focused card
            // (Future enhancement)
        }});

        // Initialize
        updateStats();
    </script>
</body>
</html>
"""

    output_html.write_text(html)
    logger.info(f"✓ Active learning interface: {output_html}")
    logger.info(f"  {len(uncertain_images)} uncertain images")
    logger.info(f"  Open: open {output_html}")

    return output_html
