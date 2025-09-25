import pandas as pd
import joblib
from pipeline.embedding import Embedder
from pipeline.scoring import Scorer
from pipeline.allocation import Allocator

class AllocationModel:
    def __init__(self):
        self.embedder = Embedder()
        self.scorer = Scorer()
        self.allocator = Allocator()
        self.fitted = False

    def fit(self, candidates_file, internships_file):
        self.candidates = pd.read_csv(candidates_file).fillna("")
        self.internships = pd.read_csv(internships_file).fillna("")

        self.candidates["Tech_Skills"] = self.candidates["Technical Skills"].apply(lambda x: self.embedder.clean_and_limit(x,5))
        self.candidates["Soft_Skills"] = self.candidates["Soft Skills"].apply(lambda x: self.embedder.clean_and_limit(x,3))
        self.candidates["Projects"] = self.candidates["Projects"].fillna("")

        self.internships["Req_Skills"] = self.internships["Skills Required"].apply(lambda x: self.embedder.clean_and_limit(x,10))
        self.internships["Responsibilities"] = self.internships["Responsibilities"].fillna("")
        self.internships["Job Description"] = self.internships["Job Description"].fillna("")

        # embeddings
        cand_tech_emb = self.embedder.encode_batch(self.candidates["Tech_Skills"].tolist())
        cand_soft_emb = self.embedder.encode_batch(self.candidates["Soft_Skills"].tolist())
        cand_proj_emb = self.embedder.encode_batch(self.candidates["Projects"].tolist())
        intern_req_emb = self.embedder.encode_batch(self.internships["Req_Skills"].tolist())
        intern_resp_emb = self.embedder.encode_batch(self.internships["Responsibilities"].tolist())
        intern_desc_emb = self.embedder.encode_batch(self.internships["Job Description"].tolist())

        # score matrix
        skill_matrix = self.scorer.compute_skill_matrix(cand_tech_emb,cand_soft_emb,cand_proj_emb,
                                                        intern_req_emb,intern_resp_emb,intern_desc_emb)
        skill_matrix = self.scorer.apply_category_adjustment(skill_matrix, self.candidates["Sector Interest"], self.internships["Category"])
        cand_locs = self.candidates[["Location Preference 1","Location Preference 2","Location Preference 3"]]
        skill_matrix = self.scorer.apply_location_adjustment(skill_matrix, cand_locs, self.internships["Location"])

        self.scores_df = pd.DataFrame(skill_matrix, index=self.candidates["Candidate ID"], columns=self.internships["Internship ID"])
        self.fitted = True

    def predict(self):
        if not self.fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.allocator.allocate(self.candidates, self.internships, self.scores_df)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
