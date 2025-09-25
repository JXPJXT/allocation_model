#!/usr/bin/env python3
import os
import sys
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import runner
try:
    from speed_optimized_runner import SpeedOptimizedSupabaseRunner
except ImportError as e:
    logger.error(f"Cannot import runner: {e}")
    sys.exit(1)

app = Flask(__name__)
CORS(app)

runner = None

@app.before_first_request
def init_runner():
    global runner
    try:
        model_path = os.environ.get('MODEL_PATH', 'models/allocation_model.pkl')
        runner = SpeedOptimizedSupabaseRunner()
        logger.info("Runner initialized")
    except Exception as e:
        logger.error(f"Runner init error: {e}")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_ready': runner is not None
    })

@app.route('/run-allocation', methods=['POST'])
def run_allocation():
    data = request.get_json() or {}
    internship_id = data.get('internship_id')
    if not internship_id:
        return jsonify({'success':False,'error':'Provide internship_id'}),400
    success = runner and runner.run_allocation_for_internship(internship_id)
    return jsonify({'success':bool(success),'internship_id':internship_id})

@app.route('/get-results/<internship_id>', methods=['GET'])
def get_results(internship_id):
    results = runner.fetch_table_data('results', {'InternshipID': internship_id})
    return jsonify({'success':True,'results':results,'count':len(results)})

@app.route('/list-internships', methods=['GET'])
def list_internships():
    ints = runner.fetch_table_data('internship')
    return jsonify({'success':True,'internships':ints,'count':len(ints)})

if __name__=='__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
