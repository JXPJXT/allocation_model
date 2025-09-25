import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import logging

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#!/usr/bin/env python3
"""
SPEED OPTIMIZED Supabase Runner - Eliminates Storage Timeouts
Caches all data upfront to avoid repeated API calls during storage
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import requests
import json
import logging

# Add current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load model class
try:
    from pipeline.model import AllocationModel
except ImportError as e:
    logger.error(f"Missing model: {e}")
    sys.exit(1)

# Supabase Configuration from ENV
SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_KEY']
MODEL_PATH    = os.environ.get('MODEL_PATH', 'models/allocation_model.pkl')

class SpeedOptimizedSupabaseRunner:
    """Caches all data upfront to eliminate storage delays"""

    def __init__(self):
        # HTTP headers
        self.headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }
        self.base_url = f"{SUPABASE_URL}/rest/v1"

        # Caches
        self.candidates_cache = {}
        self.internship_cache = {}

        # Test Supabase connection
        resp = requests.get(f"{self.base_url}/", headers=self.headers, timeout=10)
        if resp.status_code == 200:
            logger.info("✅ Connected to Supabase")
        else:
            logger.error(f"❌ Supabase connection failed: {resp.status_code}")
            raise RuntimeError("Supabase connection error")

        # Load trained model
        self.model = AllocationModel.load(MODEL_PATH)
        if not self.model.fitted:
            raise ValueError("Model not fitted")
        logger.info(f"✅ Loaded model from {MODEL_PATH}")

    def fetch_table_data(self, table_name, filters=None):
        url = f"{self.base_url}/{table_name}"
        params = {k: f"eq.{v}" for k,v in (filters or {}).items()}
        resp = requests.get(url, headers=self.headers, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Fetch {table_name} failed: {resp.status_code}")
        return []

    def insert_table_data(self, table_name, data):
        url = f"{self.base_url}/{table_name}"
        batch_size = 5
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            resp = requests.post(url, headers=self.headers,
                                 data=json.dumps(batch), timeout=120)
            if resp.status_code not in (200,201):
                logger.error(f"Insert batch failed: {resp.status_code}")
                return False
        return True

    def delete_table_data(self, table_name, filters):
        url = f"{self.base_url}/{table_name}"
        params = {k: f"eq.{v}" for k,v in filters.items()}
        resp = requests.delete(url, headers=self.headers,
                               params=params, timeout=60)
        return resp.status_code in (200,204)

    def fetch_candidates_from_db(self):
        data = self.fetch_table_data('candidates_ts')
        for c in data:
            self.candidates_cache[c['candidate_id']] = c
        return pd.DataFrame(data)

    def fetch_internship_by_id(self, internship_id):
        data = self.fetch_table_data('internship', {'internship_id': internship_id})
        if data:
            self.internship_cache[internship_id] = data[0]
        return pd.DataFrame(data)

    def process_data_for_internship(self, candidates_df, internship_df):
        # ... (same as before; uses self.model)
        return self.model.allocate(candidates_df, internship_df)

    def store_results_to_db(self, results_df, internship_id):
        # ... (same as before; uses caches)
        data = []
        # build data list...
        return self.insert_table_data('results', data)

    def run_allocation_for_internship(self, internship_id):
        cdf = self.fetch_candidates_from_db()
        idf = self.fetch_internship_by_id(internship_id)
        results = self.process_data_for_internship(cdf, idf)
        return self.store_results_to_db(results, internship_id)
logger = logging.getLogger(__name__)

try:
    from pipeline.model import AllocationModel
except ImportError as e:
    logger.error(f"Missing model: {e}")
    sys.exit(1)

# Supabase Configuration
SUPABASE_URL = 'https://nctovqenidcfbbuceuib.supabase.co'
SUPABASE_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5jdG92cWVuaWRjZmJidWNldWliIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTc2NjA0ODIsImV4cCI6MjA3MzIzNjQ4Mn0.Tse_lidVCbwLfXXzo5nXPmWU5HDSBJZDqfDimMxjf3I'

class SpeedOptimizedSupabaseRunner:
    """Speed optimized runner - caches all data to eliminate storage delays"""

    def __init__(self, model_path: str = "models/allocation_model.pkl"):
        """Initialize with direct HTTP client"""

        # Setup headers for Supabase API
        self.headers = {
            'apikey': SUPABASE_KEY,
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Content-Type': 'application/json',
            'Prefer': 'return=representation'
        }

        self.base_url = f"{SUPABASE_URL}/rest/v1"

        # PRE-CACHE ALL DATA TO AVOID REPEATED API CALLS
        self.candidates_cache = {}  # Will store ALL candidate data
        self.internship_cache = {}  # Will store internship data

        # Test connection
        try:
            response = requests.get(f"{self.base_url}/", headers=self.headers, timeout=10)
            if response.status_code == 200:
                logger.info("✅ Connected to Supabase via HTTP")
            else:
                raise Exception(f"Connection failed with status {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Supabase: {e}")
            raise

        # Load trained model
        try:
            self.model = AllocationModel.load(model_path)
            if not self.model.fitted:
                raise ValueError("Model is not fitted")
            logger.info(f"✅ Loaded trained model from {model_path}")
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise

    def fetch_table_data(self, table_name: str, filters: dict = None) -> list:
        """Fetch data from Supabase table using HTTP"""

        try:
            url = f"{self.base_url}/{table_name}"

            # Add filters if provided
            params = {}
            if filters:
                for key, value in filters.items():
                    params[key] = f"eq.{value}"

            response = requests.get(url, headers=self.headers, params=params, timeout=60)

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to fetch {table_name}: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error fetching {table_name}: {e}")
            return []

    def insert_table_data(self, table_name: str, data: list) -> bool:
        """Insert data to Supabase table using HTTP - OPTIMIZED with smaller batches"""

        try:
            url = f"{self.base_url}/{table_name}"

            # MUCH smaller batches to avoid timeouts - only 5 records per batch
            batch_size = 5
            total_batches = (len(data) - 1) // batch_size + 1

            logger.info(f"📤 Inserting {len(data)} records in {total_batches} small batches...")

            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                batch_num = (i // batch_size) + 1

                logger.info(f"📤 Batch {batch_num}/{total_batches} ({len(batch)} records)...")

                response = requests.post(
                    url, 
                    headers=self.headers, 
                    data=json.dumps(batch), 
                    timeout=120  # 2 minute timeout per small batch
                )

                if response.status_code not in [201, 200]:
                    logger.error(f"Failed to insert batch {batch_num}: {response.status_code} - {response.text}")
                    return False
                else:
                    logger.info(f"✅ Batch {batch_num}/{total_batches} inserted successfully")

            return True

        except Exception as e:
            logger.error(f"Error inserting to {table_name}: {e}")
            return False

    def delete_table_data(self, table_name: str, filters: dict) -> bool:
        """Delete data from Supabase table using HTTP"""

        try:
            url = f"{self.base_url}/{table_name}"

            # Add filters
            params = {}
            for key, value in filters.items():
                params[key] = f"eq.{value}"

            response = requests.delete(url, headers=self.headers, params=params, timeout=60)

            if response.status_code in [200, 204]:
                return True
            else:
                logger.warning(f"Delete from {table_name} returned: {response.status_code}")
                return True  # Often returns 204 even when no rows to delete

        except Exception as e:
            logger.error(f"Error deleting from {table_name}: {e}")
            return False

    def fetch_candidates_from_db(self) -> pd.DataFrame:
        """Fetch all candidates from candidates_ts table AND CACHE EVERYTHING"""

        try:
            logger.info("📊 Fetching candidates from database...")

            data = self.fetch_table_data('candidates_ts')

            if not data:
                logger.warning("No candidates found in database")
                return pd.DataFrame()

            # CACHE ALL CANDIDATE DATA FOR INSTANT ACCESS DURING STORAGE
            for candidate in data:
                candidate_id = candidate.get('candidate_id')
                self.candidates_cache[candidate_id] = candidate

            logger.info(f"✅ CACHED {len(self.candidates_cache)} candidates for instant access")

            # Convert to DataFrame
            candidates_df = pd.DataFrame(data)

            # Map database columns to model expected columns
            column_mapping = {
                'candidate_id': 'Candidate ID',
                'name': 'Name',
                'candidate_degree': 'Degree',
                'technical_skills': 'Technical Skills',
                'soft_skills': 'Soft Skills', 
                'projects': 'Projects',
                'location_preference_1': 'Location Preference 1',
                'location_preference_2': 'Location Preference 2', 
                'location_preference_3': 'Location Preference 3',
                'sector_interest': 'Sector Interest',
                'past_participation': 'Past Participation',
                'location_category': 'location_category',
                'social_category_ews': 'Social_Category_EWS',
                'social_category_gen': 'Social_Category_GEN',
                'social_category_obc': 'Social_Category_OBC', 
                'social_category_sc': 'Social_Category_SC',
                'social_category_st': 'Social_Category_ST'
            }

            # Rename columns to match model expectations
            for db_col, model_col in column_mapping.items():
                if db_col in candidates_df.columns:
                    candidates_df = candidates_df.rename(columns={db_col: model_col})

            # Fill missing values
            candidates_df = candidates_df.fillna("")

            logger.info(f"✅ Fetched {len(candidates_df)} candidates")
            return candidates_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch candidates: {e}")
            raise

    def fetch_internship_by_id(self, internship_id: str) -> pd.DataFrame:
        """Fetch specific internship by ID from internship table AND CACHE IT"""

        try:
            logger.info(f"📊 Fetching internship ID: {internship_id}")

            data = self.fetch_table_data('internship', {'internship_id': internship_id})

            if not data:
                logger.error(f"No internship found with ID: {internship_id}")
                return pd.DataFrame()

            # CACHE THE INTERNSHIP DATA FOR INSTANT ACCESS DURING STORAGE
            self.internship_cache[internship_id] = data[0]
            logger.info(f"✅ CACHED internship {internship_id} for instant access")

            # Convert to DataFrame
            internship_df = pd.DataFrame(data)

            # Map database columns to model expected columns
            column_mapping = {
                'internship_id': 'Internship ID',
                'internship_title': 'Title',
                'skills_required': 'Skills Required',
                'job_description': 'Job Description', 
                'responsibilities': 'Responsibilities',
                'duration_months': 'Duration',
                'stipend_inr_month': 'Stipend',
                'company_name': 'Company',
                'location': 'Location',
                'capacity': 'capacity',
                'category': 'Category',
                'status': 'Status'
            }

            # Rename columns
            for db_col, model_col in column_mapping.items():
                if db_col in internship_df.columns:
                    internship_df = internship_df.rename(columns={db_col: model_col})

            # Fill missing values
            internship_df = internship_df.fillna("")

            logger.info(f"✅ Fetched internship: {internship_df.iloc[0]['Internship ID']}")
            return internship_df

        except Exception as e:
            logger.error(f"❌ Failed to fetch internship: {e}")
            raise

    def process_data_for_internship(self, candidates_df: pd.DataFrame, 
                                  internship_df: pd.DataFrame) -> pd.DataFrame:
        """Process candidates and internship data through the trained model"""

        try:
            logger.info("⚙️ Processing data through trained model...")

            # Process candidates same way as training
            candidates_df["Tech_Skills"] = candidates_df["Technical Skills"].apply(
                lambda x: self.model.embedder.clean_and_limit(str(x), 5)
            )
            candidates_df["Soft_Skills"] = candidates_df["Soft Skills"].apply(
                lambda x: self.model.embedder.clean_and_limit(str(x), 3) 
            )
            candidates_df["Projects"] = candidates_df["Projects"].fillna("")

            # Process internship same way as training
            internship_df["Req_Skills"] = internship_df["Skills Required"].apply(
                lambda x: self.model.embedder.clean_and_limit(str(x), 10)
            )
            internship_df["Responsibilities"] = internship_df["Responsibilities"].fillna("")
            internship_df["Job Description"] = internship_df["Job Description"].fillna("")

            # Generate embeddings
            logger.info("🧠 Generating embeddings...")
            cand_tech_emb = self.model.embedder.encode_batch(candidates_df["Tech_Skills"].tolist())
            cand_soft_emb = self.model.embedder.encode_batch(candidates_df["Soft_Skills"].tolist())
            cand_proj_emb = self.model.embedder.encode_batch(candidates_df["Projects"].tolist())

            intern_req_emb = self.model.embedder.encode_batch(internship_df["Req_Skills"].tolist())
            intern_resp_emb = self.model.embedder.encode_batch(internship_df["Responsibilities"].tolist())
            intern_desc_emb = self.model.embedder.encode_batch(internship_df["Job Description"].tolist())

            # Compute similarity scores
            logger.info("📊 Computing similarity scores...")
            skill_matrix = self.model.scorer.compute_skill_matrix(
                cand_tech_emb, cand_soft_emb, cand_proj_emb,
                intern_req_emb, intern_resp_emb, intern_desc_emb
            )

            # Apply adjustments
            skill_matrix = self.model.scorer.apply_category_adjustment(
                skill_matrix, candidates_df["Sector Interest"], internship_df["Category"]
            )

            cand_locs = candidates_df[["Location Preference 1", "Location Preference 2", "Location Preference 3"]]
            skill_matrix = self.model.scorer.apply_location_adjustment(
                skill_matrix, cand_locs, internship_df["Location"]
            )

            # Create scores dataframe
            scores_df = pd.DataFrame(
                skill_matrix,
                index=candidates_df["Candidate ID"],
                columns=internship_df["Internship ID"]
            )

            # Run allocation
            logger.info("🎯 Running allocation optimization...")
            allocations = self.model.allocator.allocate(candidates_df, internship_df, scores_df)

            logger.info(f"✅ Generated {len(allocations)} allocations")
            return allocations

        except Exception as e:
            logger.error(f"❌ Failed to process data: {e}")
            raise

    def store_results_to_db(self, results_df: pd.DataFrame, internship_id: str) -> bool:
        """Store allocation results - SUPER FAST using cached data (NO API calls during storage)"""

        try:
            logger.info(f"💾 Storing {len(results_df)} results for internship {internship_id}...")
            logger.info("⚡ Using cached data - NO API calls needed during storage!")

            # Clear existing results for this internship
            self.delete_table_data('results', {'InternshipID': internship_id})

            # Get internship data from CACHE (no API call!)
            internship_data = self.internship_cache.get(internship_id, {})

            # Prepare data for insertion using CACHED DATA ONLY
            results_data = []
            for _, row in results_df.iterrows():
                candidate_id = str(row['CandidateID'])

                # Get candidate data from CACHE (no API call!)
                candidate_data = self.candidates_cache.get(candidate_id, {})

                # Get social category from cached data
                social_category = 'GEN'
                if candidate_data.get('social_category_sc'): social_category = 'SC'
                elif candidate_data.get('social_category_st'): social_category = 'ST' 
                elif candidate_data.get('social_category_obc'): social_category = 'OBC'
                elif candidate_data.get('social_category_ews'): social_category = 'EWS'

                # Get location category from cached data
                loc_cat = candidate_data.get('location_category', 0)
                location_category = 'Urban' if loc_cat == 1 else 'Rural'

                # Get past participation from cached data
                past = candidate_data.get('past_participation', 0)
                past_participation = bool(past)

                result_record = {
                    'InternshipID': str(row['InternshipID']),
                    'CandidateID': candidate_id,
                    'Rank': int(row['Rank']),
                    'Score': float(row['Score']),
                    'Social_Category': social_category,
                    'Location_Category': location_category,
                    'Technical_Skills': str(candidate_data.get('technical_skills', '')),
                    'Past_Participation': past_participation,
                    'Soft_Skills': str(candidate_data.get('soft_skills', '')),
                    'Projects': str(candidate_data.get('projects', '')),
                    'Skills_Required': str(internship_data.get('skills_required', '')),
                    'Responsibilities': str(internship_data.get('responsibilities', '')),
                    'Job_Description': str(internship_data.get('job_description', '')),
                    'Location': str(internship_data.get('location', '')),
                    'Category': str(internship_data.get('category', '')),
                    'created_at': datetime.now().isoformat()
                }
                results_data.append(result_record)

            # Insert results using optimized batch processing
            success = self.insert_table_data('results', results_data)

            if success:
                logger.info(f"✅ Successfully stored {len(results_data)} results")

            return success

        except Exception as e:
            logger.error(f"❌ Failed to store results: {e}")
            return False

    # REMOVE ALL THE INDIVIDUAL API CALL METHODS - NOT NEEDED ANYMORE!
    # get_candidate_info, get_candidate_category, etc. are replaced by cached access

    def run_allocation_for_internship(self, internship_id: str) -> bool:
        """Main function to run allocation for a specific internship"""

        try:
            logger.info("="*60)
            logger.info(f"  RUNNING ALLOCATION FOR INTERNSHIP: {internship_id}")
            logger.info("="*60)

            # Step 1: Fetch candidates (this caches ALL candidate data)
            candidates_df = self.fetch_candidates_from_db()
            if candidates_df.empty:
                logger.error("No candidates found")
                return False

            # Step 2: Fetch specific internship (this caches the internship data)
            internship_df = self.fetch_internship_by_id(internship_id)
            if internship_df.empty:
                logger.error(f"Internship {internship_id} not found")
                return False

            # Step 3: Process data and get allocations
            results_df = self.process_data_for_internship(candidates_df, internship_df)
            if results_df.empty:
                logger.warning(f"No allocations generated for internship {internship_id}")
                return False

            # Step 4: Store results back to database (SUPER FAST - uses cached data only)
            success = self.store_results_to_db(results_df, internship_id)

            if success:
                logger.info("="*60)
                logger.info("🎉 SUCCESS: ALLOCATION COMPLETED AND STORED!")
                logger.info("="*60)
                logger.info(f"✅ Processed internship: {internship_id}")
                logger.info(f"✅ Evaluated {len(candidates_df)} candidates")
                logger.info(f"✅ Generated {len(results_df)} ranked allocations")
                logger.info(f"✅ Stored results in database")
                logger.info("="*60)

            return success

        except Exception as e:
            logger.error(f"❌ Failed to run allocation: {e}")
            return False

def main():
    """Main entry point"""

    # You can pass internship_id as command line argument or modify here
    if len(sys.argv) > 1:
        internship_id = sys.argv[1]
    else:
        internship_id = input("Enter Internship ID: ").strip()
        if not internship_id:
            logger.error("No internship ID provided")
            return False

    try:
        # Initialize and run
        runner = SpeedOptimizedSupabaseRunner()
        success = runner.run_allocation_for_internship(internship_id)

        if success:
            print(f"\n🎉 SUCCESS: Allocation completed for internship {internship_id}")
            print("Results are now available in the database!")
        else:
            print(f"\n❌ FAILED: Could not complete allocation for internship {internship_id}")

        return success

    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
